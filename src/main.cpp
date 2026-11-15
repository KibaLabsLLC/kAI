// kibavoice - local LLM + speech-to-text + text-to-speech, with a floating UI.
//
//   kibavoice ui [--background] [--listen]   (opens kibavoice-ui, the voice-only window)
//   kibavoice fetch | doctor | install-desktop [--autostart]
//   kibavoice install-service [--no-start] | uninstall-service   (always-on systemd user daemon)
//   kibavoice ask <text...>  [--no-speak] [--no-notify]
//   kibavoice listen         [--no-speak] [--no-notify]
//   kibavoice transcribe <wav> [--wake]   (prints the text; used by the UI)
//   kibavoice say <text...>
//   kibavoice history | show <id> | replay <id>
//
// LLM : llama.cpp (linked)        STT : whisper-cli (child process)
// TTS : Piper (child process)     Audio I/O : arecord + paplay/pw-play/aplay
// Child processes are started with execvp and an argv vector, never a shell,
// so text from the model or the microphone cannot inject commands.

#include "ggml-backend.h"
#include "llama.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------- config ---
static std::string env_or(const char* k, const std::string& d) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : d;
}

static const char* kDefaultLlmUrl =
    "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
static const char* kDefaultSttUrl =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
static const char* kDefaultVoice = "en_US-lessac-medium";
static const char* kPiperBase =
    "https://github.com/rhasspy/piper/releases/download/v1.2.0/";

static const char* kSystemPrompt =
    "You are a concise voice assistant inside a desktop launcher. Answer in at most "
    "three short sentences of plain spoken English. No markdown, no lists, no emoji."
    "Your name is Kiba and you were made by Kiba Labs, LLC for the KibaOS Operating system, your character is an orb with raccoon ears and tail";

static fs::path home() { return env_or("HOME", "/tmp"); }
static fs::path data_dir() {
    fs::path base = env_or("XDG_DATA_HOME", (home() / ".local/share").string());
    return base / "kibavoice";
}
static bool debug() { return std::getenv("KIBAVOICE_DEBUG") != nullptr; }

// -------------------------------------------------------------- process ----
struct RunOpts {
    std::string in;                  // fed to stdin when non-empty
    std::string* out = nullptr;      // captures stdout when set
    bool quiet = false;              // discard stderr
    std::vector<std::string> env;    // extra KEY=VALUE
};

static int run(const std::vector<std::string>& argv, const RunOpts& o = {}) {
    int inp[2] = {-1, -1}, outp[2] = {-1, -1};
    if (!o.in.empty() && pipe(inp) != 0) return -1;
    if (o.out && pipe(outp) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        if (inp[0] >= 0) { dup2(inp[0], 0); close(inp[0]); close(inp[1]); }
        else { int n = open("/dev/null", O_RDONLY); dup2(n, 0); close(n); }
        if (outp[1] >= 0) { dup2(outp[1], 1); close(outp[0]); close(outp[1]); }
        if (o.quiet && !debug()) { int n = open("/dev/null", O_WRONLY); dup2(n, 2); close(n); }
        for (auto& e : o.env) {
            auto eq = e.find('=');
            if (eq != std::string::npos) setenv(e.substr(0, eq).c_str(), e.substr(eq + 1).c_str(), 1);
        }
        std::vector<char*> a;
        for (auto& s : argv) a.push_back(const_cast<char*>(s.c_str()));
        a.push_back(nullptr);
        execvp(a[0], a.data());
        _exit(127);
    }
    if (inp[0] >= 0) {
        close(inp[0]);
        size_t off = 0;
        while (off < o.in.size()) {
            ssize_t w = write(inp[1], o.in.data() + off, o.in.size() - off);
            if (w <= 0) break;
            off += (size_t)w;
        }
        close(inp[1]);
    }
    if (outp[0] >= 0) {
        close(outp[1]);
        char buf[4096];
        ssize_t n;
        while ((n = read(outp[0], buf, sizeof buf)) > 0) o.out->append(buf, (size_t)n);
        close(outp[0]);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

static std::string which(const std::string& name) {
    if (name.find('/') != std::string::npos) return fs::exists(name) ? name : "";
    std::stringstream ss(env_or("PATH", "/usr/bin:/bin"));
    std::string d;
    while (std::getline(ss, d, ':')) {
        fs::path p = fs::path(d) / name;
        if (!d.empty() && access(p.c_str(), X_OK) == 0) return p.string();
    }
    return "";
}

static fs::path exe_path() {
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path("kibavoice") : p;
}

// ------------------------------------------------------------- utilities ---
static std::string trim(std::string s) {
    auto sp = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && sp(s.back())) s.pop_back();
    size_t i = 0;
    while (i < s.size() && sp(s[i])) ++i;
    return s.substr(i);
}
static std::string oneline(std::string s, size_t max) {
    for (auto& c : s) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    s = trim(s);
    if (s.size() > max) { s.resize(max); s += "..."; }
    return s;
}
static std::string join_args(int argc, char** argv, int from, bool* nospeak, bool* nonotify) {
    std::string t;
    for (int i = from; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-speak") { if (nospeak) *nospeak = true; continue; }
        if (a == "--no-notify") { if (nonotify) *nonotify = true; continue; }
        if (!t.empty()) t += ' ';
        t += a;
    }
    return trim(t);
}
static void notify(const std::string& title, const std::string& body) {
    if (which("notify-send").empty()) return;
    run({"notify-send", "-a", "kibavoice", "-t", "15000", oneline(title, 80), body}, {"", nullptr, true, {}});
}
static std::string clean_for_speech(std::string s) {
    std::string o;
    for (char c : s) if (c != '*' && c != '`' && c != '#' && c != '_') o += c;
    return trim(o);
}

// ------------------------------------------------------------- download ----
static bool download(const std::string& url, const fs::path& dest, uintmax_t min_bytes) {
    std::error_code ec;
    if (fs::exists(dest, ec) && fs::file_size(dest, ec) >= min_bytes) return true;
    fs::create_directories(dest.parent_path(), ec);
    fs::path part = dest; part += ".part";
    std::cerr << "kibavoice: downloading " << dest.filename().string() << "\n";
    int rc = run({"curl", "-fL", "--retry", "3", "--connect-timeout", "20", "-C", "-", "-o", part.string(), url});
    // 33 = server refused the resume range because the .part file is already complete
    if (rc != 0 && !(rc == 33 && fs::exists(part, ec))) {
        std::cerr << "kibavoice: download failed (" << url << ")\n";
        return false;
    }
    if (!fs::exists(part, ec) || fs::file_size(part, ec) < min_bytes) {
        std::cerr << "kibavoice: " << dest.filename().string() << " is smaller than expected, discarding\n";
        fs::remove(part, ec);
        return false;
    }
    fs::rename(part, dest, ec);
    return !ec;
}

struct Assets {
    fs::path llm, stt, piper_dir, piper_bin, voice, voice_json;
    std::string llm_url, stt_url, voice_id;
};

static Assets assets() {
    Assets a;
    fs::path d = data_dir();
    a.llm_url = env_or("KIBAVOICE_LLM_URL", kDefaultLlmUrl);
    a.stt_url = env_or("KIBAVOICE_STT_URL", kDefaultSttUrl);
    a.voice_id = env_or("KIBAVOICE_VOICE", kDefaultVoice);
    a.llm = d / "models" / fs::path(a.llm_url).filename();
    a.stt = d / "models" / fs::path(a.stt_url).filename();
    a.piper_dir = d / "piper";
    a.piper_bin = a.piper_dir / "piper";
    a.voice = d / "voices" / (a.voice_id + ".onnx");
    a.voice_json = d / "voices" / (a.voice_id + ".onnx.json");
    return a;
}

struct Lock {
    int fd = -1;
    Lock() {
        std::error_code ec;
        fs::create_directories(data_dir(), ec);
        fd = open((data_dir() / "fetch.lock").c_str(), O_CREAT | O_RDWR, 0644);
        if (fd >= 0) flock(fd, LOCK_EX);
    }
    ~Lock() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
};

static bool ensure_llm(const Assets& a) {
    Lock l; return download(a.llm_url, a.llm, 100ull << 20);
}
static bool ensure_stt(const Assets& a) {
    Lock l; return download(a.stt_url, a.stt, 20ull << 20);
}
static bool ensure_tts(const Assets& a) {
    Lock l;
    if (access(a.piper_bin.c_str(), X_OK) != 0) {
        std::string m;
        run({"uname", "-m"}, {"", &m, true, {}});
        m = trim(m);
        std::string tar = (m == "aarch64" || m == "arm64") ? "piper_arm64.tar.gz"
                        : (m == "x86_64") ? "piper_amd64.tar.gz" : "";
        if (tar.empty()) { std::cerr << "kibavoice: no Piper build for " << m << "\n"; return false; }
        fs::path tgz = data_dir() / "downloads" / tar;
        if (!download(std::string(kPiperBase) + tar, tgz, 1ull << 20)) return false;
        if (run({"tar", "-xzf", tgz.string(), "-C", data_dir().string()}) != 0) return false;
    }
    // voice id "en_US-lessac-medium" -> en/en_US/lessac/medium/
    std::vector<std::string> p;
    std::stringstream ss(a.voice_id);
    for (std::string s; std::getline(ss, s, '-');) p.push_back(s);
    if (p.size() != 3 || p[0].size() < 2) { std::cerr << "kibavoice: bad KIBAVOICE_VOICE id\n"; return false; }
    // Pinned to the v1.0.0 tag (what Piper's own docs use) so a reshuffle of `main` can't break downloads.
    std::string base = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/" + p[0].substr(0, 2) + "/" +
                       p[0] + "/" + p[1] + "/" + p[2] + "/" + a.voice_id;
    return download(base + ".onnx", a.voice, 1ull << 20) && download(base + ".onnx.json", a.voice_json, 50);
}

// ------------------------------------------------------------------ LLM ----
static void quiet_log(ggml_log_level lvl, const char* text, void*) {
    if (debug() || lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static std::string generate(const fs::path& model_path, const std::string& user, int max_new) {
    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return "";
    const llama_vocab* vocab = llama_model_get_vocab(model);

    int threads = (int)std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return ""; }

    // Build the prompt with the model's own chat template; fall back to ChatML.
    std::string sys = env_or("KIBAVOICE_SYSTEM", kSystemPrompt);
    llama_chat_message msgs[2] = {{"system", sys.c_str()}, {"user", user.c_str()}};
    const char* tmpl = llama_model_chat_template(model, nullptr);
    std::string prompt;
    if (tmpl) {
        std::vector<char> buf(sys.size() + user.size() * 2 + 2048);
        int n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
        if (n > (int)buf.size()) {
            buf.resize((size_t)n);
            n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int)buf.size());
        }
        if (n > 0) prompt.assign(buf.data(), (size_t)n);
    }
    if (prompt.empty())
        prompt = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + user +
                 "<|im_end|>\n<|im_start|>assistant\n";

    int nt = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks((size_t)std::max(nt, 1));
    nt = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(), true, true);
    std::string out;
    if (nt <= 0 || nt + max_new > (int)cp.n_ctx) {
        std::cerr << "kibavoice: prompt too long for the context window\n";
    } else {
        toks.resize((size_t)nt);
        llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.6f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)time(nullptr)));

        bool ok = true;
        for (int i = 0; i < nt && ok; i += (int)cp.n_batch) {
            int n = std::min((int)cp.n_batch, nt - i);
            ok = llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) == 0;
        }
        for (int i = 0; ok && i < max_new; ++i) {
            llama_token t = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, t)) break;
            char piece[256];
            int n = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
            if (n > 0) out.append(piece, (size_t)n);
            ok = llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0;
        }
        llama_sampler_free(smpl);
    }
    llama_free(ctx);
    llama_model_free(model);
    return trim(out);
}

// ---------------------------------------------------------------- audio ----
static void write_wav(const fs::path& p, const std::vector<int16_t>& pcm, uint32_t rate) {
    std::ofstream f(p, std::ios::binary);
    uint32_t data = (uint32_t)(pcm.size() * 2), riff = 36 + data, fmt = 16, br = rate * 2;
    uint16_t pcmfmt = 1, ch = 1, align = 2, bits = 16;
    f.write("RIFF", 4); f.write((char*)&riff, 4); f.write("WAVEfmt ", 8);
    f.write((char*)&fmt, 4); f.write((char*)&pcmfmt, 2); f.write((char*)&ch, 2);
    f.write((char*)&rate, 4); f.write((char*)&br, 4); f.write((char*)&align, 2);
    f.write((char*)&bits, 2); f.write("data", 4); f.write((char*)&data, 4);
    f.write((const char*)pcm.data(), (std::streamsize)data);
}

// Records 16 kHz mono until ~1 s of silence follows speech (max 15 s).
static bool record_utterance(const fs::path& wav) {
    std::string cmd = env_or("KIBAVOICE_REC_CMD", "arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null");
    FILE* p = popen(cmd.c_str(), "r");   // command is from env/constant, not from user text
    if (!p) return false;
    const int frame = 320;               // 20 ms
    std::vector<int16_t> all, f((size_t)frame);
    double noise = 0; int calib = 0, loud = 0, quiet = 0; bool started = false;
    double thr = 600;
    for (int n = 0; n < 750; ++n) {      // 750 * 20 ms = 15 s
        if (fread(f.data(), 2, (size_t)frame, p) != (size_t)frame) break;
        all.insert(all.end(), f.begin(), f.end());
        double e = 0;
        for (int16_t s : f) e += (double)s * s;
        double rms = std::sqrt(e / frame);
        if (calib < 15) { noise += rms / 15; if (++calib == 15) thr = std::max(500.0, noise * 3.0); continue; }
        if (!started) {
            loud = rms > thr ? loud + 1 : 0;
            if (loud >= 3) started = true;
            else if (n > 300) break;     // 6 s with no speech
        } else {
            quiet = rms < thr ? quiet + 1 : 0;
            if (quiet >= 50) break;      // 1 s of silence
        }
    }
    pclose(p);
    if (!started) return false;
    write_wav(wav, all, 16000);
    return true;
}

static std::string whisper_bin() {
    fs::path d = exe_path().parent_path();
    for (fs::path c : {d / "whisper-cli", d / ".." / "libexec" / "kibavoice" / "whisper-cli"})
        if (access(c.c_str(), X_OK) == 0) return c.string();
    return which("whisper-cli");
}

// wake=true: bias Whisper toward "Hey Kiba" and decode greedily (fast, for always-on listening).
static std::string transcribe(const Assets& a, const fs::path& wav, bool wake = false) {
    std::string bin = whisper_bin();
    if (bin.empty()) { std::cerr << "kibavoice: whisper-cli not found (build with KIBAVOICE_BUILD_WHISPER=ON)\n"; return ""; }
    std::string out;
    int thr = (int)std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    std::vector<std::string> cmd{bin, "-m", a.stt.string(), "-f", wav.string(), "-nt", "-np", "-t", std::to_string(thr)};
    if (wake) for (const char* x : {"--prompt", "Hey Kiba.", "-bs", "1", "-bo", "1"}) cmd.push_back(x);
    run(cmd, {"", &out, true, {}});
    out = oneline(out, 1000);
    if (out.find("[BLANK_AUDIO]") != std::string::npos) return "";
    return out;
}

static void speak(const Assets& a, const std::string& text) {
    std::string t = clean_for_speech(text);
    if (t.empty()) return;
    fs::path wav = data_dir() / "tmp-speech.wav";
    std::string ld = env_or("LD_LIBRARY_PATH", "");
    int rc = run({a.piper_bin.string(), "--model", a.voice.string(), "--espeak_data",
                  (a.piper_dir / "espeak-ng-data").string(), "--output_file", wav.string()},
                 {t, nullptr, true, {"LD_LIBRARY_PATH=" + a.piper_dir.string() + (ld.empty() ? "" : ":" + ld)}});
    if (rc != 0) { std::cerr << "kibavoice: piper failed\n"; return; }
    for (auto& player : std::vector<std::vector<std::string>>{{"paplay"}, {"pw-play"}, {"aplay", "-q"}}) {
        if (which(player[0]).empty()) continue;
        auto argv = player; argv.push_back(wav.string());
        run(argv, {"", nullptr, true, {}});
        return;
    }
    std::cerr << "kibavoice: no audio player (paplay/pw-play/aplay)\n";
}

// -------------------------------------------------------------- history ----
static bool valid_id(const std::string& id) {
    return !id.empty() && id.size() < 24 && std::all_of(id.begin(), id.end(), ::isdigit);
}
static void save_history(const std::string& q, const std::string& ans) {
    fs::path h = data_dir() / "history";
    std::error_code ec;
    fs::create_directories(h, ec);
    std::string id = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    std::ofstream(h / (id + ".q")) << q;
    std::ofstream(h / (id + ".a")) << ans;
    std::ofstream(data_dir() / "history.tsv", std::ios::app)
        << id << '\t' << oneline(q, 80) << '\t' << oneline(ans, 120) << '\n';
}
static std::string slurp(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// ------------------------------------------------------------- commands ----
static int do_ask(const std::string& q, bool nospeak, bool nonotify) {
    Assets a = assets();
    if (q.empty()) { std::cerr << "usage: kibavoice ask <question>\n"; return 2; }
    if (!fs::exists(a.llm) && !nonotify) notify("kibavoice", "First run: downloading the language model...");
    if (!ensure_llm(a)) return 1;
    std::string ans = generate(a.llm, q, 160);
    if (ans.empty()) { std::cerr << "kibavoice: model produced no answer\n"; return 1; }
    save_history(q, ans);
    std::cout << ans << "\n";
    if (!nonotify) notify(q, ans);
    if (!nospeak && ensure_tts(a)) speak(a, ans);
    return 0;
}

static int do_listen(bool nospeak, bool nonotify) {
    Assets a = assets();
    if (!nonotify && (!fs::exists(a.stt) || !fs::exists(a.llm)))
        notify("kibavoice", "First run: downloading speech and language models...");
    if (!ensure_stt(a) || !ensure_llm(a)) return 1;
    if (which("arecord").empty() && !std::getenv("KIBAVOICE_REC_CMD")) {
        std::cerr << "kibavoice: arecord not found (install alsa-utils)\n"; return 1;
    }
    if (!nonotify) notify("kibavoice", "Listening...");
    fs::path wav = data_dir() / "tmp-input.wav";
    std::error_code ec;
    fs::create_directories(data_dir(), ec);
    if (!record_utterance(wav)) {
        if (!nonotify) notify("kibavoice", "Didn't hear anything.");
        return 1;
    }
    std::string q = transcribe(a, wav);
    fs::remove(wav, ec);
    if (q.empty()) { if (!nonotify) notify("kibavoice", "Couldn't make that out."); return 1; }
    return do_ask(q, nospeak, nonotify);
}

static int do_say(const std::string& t) {
    Assets a = assets();
    if (t.empty() || !ensure_tts(a)) return 1;
    speak(a, t);
    return 0;
}

static int do_fetch() {
    Assets a = assets();
    bool ok = ensure_llm(a);
    ok = ensure_stt(a) && ok;
    ok = ensure_tts(a) && ok;
    std::cerr << (ok ? "kibavoice: all models ready in " : "kibavoice: some downloads failed; data dir is ")
              << data_dir().string() << "\n";
    return ok ? 0 : 1;
}

static fs::path ui_bin();
static bool systemctl_user(const std::vector<std::string>& args, std::string* out);
static int do_doctor() {
    Assets a = assets();
    auto line = [](const std::string& n, bool ok, const std::string& extra = "") {
        std::cout << (ok ? "[ok]      " : "[missing] ") << n << (extra.empty() ? "" : "  " + extra) << "\n";
    };
    line("llm model", fs::exists(a.llm), a.llm.string());
    line("stt model", fs::exists(a.stt), a.stt.string());
    line("piper", access(a.piper_bin.c_str(), X_OK) == 0, a.piper_bin.string());
    line("voice", fs::exists(a.voice), a.voice_id);
    line("whisper-cli", !whisper_bin().empty(), whisper_bin());
    line("kibavoice-ui (floating window)", !ui_bin().empty(), ui_bin().string());
    for (auto t : {"curl", "tar", "arecord"}) line(t, !which(t).empty());
    line("audio player", !which("paplay").empty() || !which("pw-play").empty() || !which("aplay").empty());
    line("notify-send (optional)", !which("notify-send").empty());
    if (!which("systemctl").empty()) {
        std::string st;
        systemctl_user({"is-active", "kibavoice.service"}, &st);
        st = trim(st);
        std::string en;
        systemctl_user({"is-enabled", "kibavoice.service"}, &en);
        en = trim(en);
        const bool installed = !en.empty() && en != "not-found";
        line("daemon (systemd user service)", st == "active",
             installed ? ("state=" + (st.empty() ? "unknown" : st) + ", " + en)
                       : "not installed (kibavoice install-service)");
    }
    return 0;
}

static fs::path ui_bin() {
    fs::path c = exe_path().parent_path() / "kibavoice-ui";
    if (access(c.c_str(), X_OK) == 0) return c;
    return which("kibavoice-ui");
}

static int do_ui(int argc, char** argv) {
    fs::path ui = ui_bin();
    if (ui.empty()) {
        std::cerr << "kibavoice: kibavoice-ui not found (build with Qt6: -DKIBAVOICE_BUILD_UI=ON)\n";
        return 1;
    }
    std::vector<std::string> a{ui.string()};
    for (int i = 2; i < argc; ++i) a.push_back(argv[i]);
    std::vector<char*> v;
    for (auto& x : a) v.push_back(const_cast<char*>(x.c_str()));
    v.push_back(nullptr);
    execv(v[0], v.data());
    std::cerr << "kibavoice: could not start " << ui << "\n";
    return 1;
}

static std::string desktop_quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '`' || c == '$' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

// Menu entry + app icon, so the window can be launched from the desktop menu or a hotkey.
static int do_install_desktop(bool autostart) {
    fs::path ui = ui_bin();
    if (ui.empty()) { std::cerr << "kibavoice: kibavoice-ui not found\n"; return 1; }
    std::error_code ec;
    fs::path share = fs::path(env_or("XDG_DATA_HOME", (home() / ".local/share").string()));
    fs::path icon_src = exe_path().parent_path() / ".." / "share" / "kibavoice" / "kibavoice.png";
    fs::path icon_dir = share / "icons/hicolor/256x256/apps";
    fs::create_directories(icon_dir, ec);
    if (fs::exists(icon_src, ec)) fs::copy_file(icon_src, icon_dir / "kibavoice.png", fs::copy_options::overwrite_existing, ec);
    fs::path dir = share / "applications";
    fs::create_directories(dir, ec);
    fs::path file = dir / "kibavoice.desktop";
    std::ofstream f(file);
    if (!f) { std::cerr << "kibavoice: cannot write " << file << "\n"; return 1; }
    const std::string exec = desktop_quote(ui.string());
    f << "[Desktop Entry]\nType=Application\nName=Kiba\nGenericName=Voice Assistant\n"
      << "Comment=Local, offline voice assistant\nKeywords=voice;assistant;kiba;speech;\n"
      << "Exec=" << exec << "\nIcon=kibavoice\nTerminal=false\nStartupNotify=false\n"
      << "Categories=Utility;AudioVideo;\nStartupWMClass=kibavoice\nActions=Listen;\n\n"
      << "[Desktop Action Listen]\nName=Talk to Kiba\nExec=" << exec << " --listen\n";
    f.close();
    std::cout << "wrote " << file.string() << "\n";
    // best effort: make the menu entry and icon show up without a re-login
    RunOpts q{"", nullptr, true, {}};
    if (!which("update-desktop-database").empty()) run({"update-desktop-database", dir.string()}, q);
    if (!which("gtk-update-icon-cache").empty())
        run({"gtk-update-icon-cache", "-q", "-t", "-f", (share / "icons/hicolor").string()}, q);
    if (autostart) {   // start hidden at login; the window appears when you say "Hey Kiba"
        fs::path adir = fs::path(env_or("XDG_CONFIG_HOME", (home() / ".config").string())) / "autostart";
        fs::create_directories(adir, ec);
        std::ofstream a(adir / "kibavoice.desktop");
        if (!a) { std::cerr << "kibavoice: cannot write autostart entry\n"; return 1; }
        a << "[Desktop Entry]\nType=Application\nName=Kiba\nComment=Listens for \"Hey Kiba\"\n"
          << "Exec=" << exec << " --background\nIcon=kibavoice\nTerminal=false\nX-GNOME-Autostart-enabled=true\n";
        std::cout << "wrote " << (adir / "kibavoice.desktop").string() << " (starts hidden at login)\n";
    }
    return 0;
}

// systemd unit values: double-quoted, with backslash, quote, % and $ escaped.
static std::string systemd_quote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '%' || c == '$') o += c;   // doubled: %% and $$
        o += c;
    }
    return o + "\"";
}

static fs::path config_home() {
    return fs::path(env_or("XDG_CONFIG_HOME", (home() / ".config").string()));
}

static bool systemctl_user(const std::vector<std::string>& args, std::string* out) {
    std::vector<std::string> a{"systemctl", "--user"};
    a.insert(a.end(), args.begin(), args.end());
    RunOpts o; o.quiet = true; o.out = out;
    return run(a, o) == 0;
}

// Always-on mode: a systemd *user* service running `kibavoice-ui --background`.
// It must be a user service (not a system one) because the microphone, audio output and the
// popup window all belong to the logged-in desktop session.
static int do_install_service(bool start) {
    fs::path ui = ui_bin();
    if (ui.empty()) { std::cerr << "kibavoice: kibavoice-ui not found (build with Qt6)\n"; return 1; }
    if (which("systemctl").empty()) {
        std::cerr << "kibavoice: systemctl not found; use `kibavoice install-desktop --autostart` instead\n";
        return 1;
    }
    std::error_code ec;
    fs::path dir = config_home() / "systemd" / "user";
    fs::create_directories(dir, ec);
    fs::path unit = dir / "kibavoice.service";
    std::ofstream f(unit);
    if (!f) { std::cerr << "kibavoice: cannot write " << unit << "\n"; return 1; }
    f << "[Unit]\n"
         "Description=Kiba voice assistant (listens for \"Hey Kiba\")\n"
         "After=graphical-session.target pipewire.service pipewire-pulse.service sound.target\n"
         "PartOf=graphical-session.target\n"
         "StartLimitIntervalSec=120\n"
         "StartLimitBurst=8\n\n"
         "[Service]\n"
         "Type=simple\n"
         "ExecStart=" << systemd_quote(ui.string()) << " --background\n"
         "# Restart after crashes and kills. A clean exit (tray Quit, `kibavoice-ui --quit`) stays stopped.\n"
         "Restart=on-failure\n"
         "RestartSec=3\n"
         "Nice=5\n"
         "Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin\n"
         "\n"
         "[Install]\n"
         "WantedBy=graphical-session.target\n";
    f.close();
    std::cout << "wrote " << unit.string() << "\n";

    // Hand the running session's display variables to the user manager (no-op if unset).
    systemctl_user({"import-environment", "DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_SESSION_TYPE"}, nullptr);
    if (!systemctl_user({"daemon-reload"}, nullptr)) {
        std::cerr << "kibavoice: `systemctl --user` is not available in this session\n"
                     "           (unit written; enable it later with: systemctl --user enable --now kibavoice)\n";
        return 1;
    }
    bool ok = systemctl_user(start ? std::vector<std::string>{"enable", "--now", "kibavoice.service"}
                                   : std::vector<std::string>{"enable", "kibavoice.service"}, nullptr);
    if (!ok) { std::cerr << "kibavoice: could not enable the service\n"; return 1; }

    // The service replaces the plain autostart entry; two launchers would just race.
    // Only drop it when the session really starts graphical-session.target (plain i3/sway
    // setups often don't, and would then lose autostart entirely).
    fs::path legacy = config_home() / "autostart" / "kibavoice.desktop";
    if (fs::exists(legacy, ec)) {
        if (systemctl_user({"is-active", "--quiet", "graphical-session.target"}, nullptr)) {
            fs::remove(legacy, ec);
            std::cout << "removed old autostart entry (the service replaces it)\n";
        } else {
            std::cout << "kept " << legacy.string() << " (this session does not activate graphical-session.target)\n";
        }
    }
    std::cout << (start ? "kibavoice daemon enabled and started.\n" : "kibavoice daemon enabled (starts at next login).\n")
              << "  status : systemctl --user status kibavoice\n"
                 "  logs   : journalctl --user -u kibavoice -f\n"
                 "  stop   : kibavoice-ui --quit   (or: systemctl --user stop kibavoice)\n";
    return 0;
}

static int do_uninstall_service() {
    std::error_code ec;
    if (!which("systemctl").empty()) systemctl_user({"disable", "--now", "kibavoice.service"}, nullptr);
    fs::path unit = config_home() / "systemd" / "user" / "kibavoice.service";
    if (fs::remove(unit, ec)) std::cout << "removed " << unit.string() << "\n";
    if (!which("systemctl").empty()) systemctl_user({"daemon-reload"}, nullptr);
    std::cout << "kibavoice daemon removed.\n";
    return 0;
}

static int do_transcribe(const std::string& wav, bool wake) {
    Assets a = assets();
    std::error_code ec;
    if (wav.empty() || !fs::exists(wav, ec)) { std::cerr << "usage: kibavoice transcribe <file.wav>\n"; return 2; }
    if (!ensure_stt(a)) return 1;
    std::string q = transcribe(a, wav, wake);
    if (q.empty()) return 1;
    std::cout << q << "\n";
    return 0;
}

static int do_history() {
    std::ifstream f(data_dir() / "history.tsv");
    std::vector<std::string> lines;
    for (std::string l; std::getline(f, l);) lines.push_back(l);
    size_t from = lines.size() > 50 ? lines.size() - 50 : 0;
    for (size_t i = lines.size(); i-- > from;) std::cout << lines[i] << "\n";
    return 0;
}

int main(int argc, char** argv) {
    std::string cmd = argc >= 2 ? argv[1] : "help";
    bool nospeak = false, nonotify = false;
    if (cmd == "ask")   return do_ask(join_args(argc, argv, 2, &nospeak, &nonotify), nospeak, nonotify);
    if (cmd == "listen") { join_args(argc, argv, 2, &nospeak, &nonotify); return do_listen(nospeak, nonotify); }
    if (cmd == "say")   return do_say(join_args(argc, argv, 2, nullptr, nullptr));
    if (cmd == "fetch") return do_fetch();
    if (cmd == "doctor") return do_doctor();
    if (cmd == "ui") return do_ui(argc, argv);
    if (cmd == "install-service") return do_install_service(!(argc > 2 && std::string(argv[2]) == "--no-start"));
    if (cmd == "uninstall-service") return do_uninstall_service();
    if (cmd == "install-desktop") return do_install_desktop(argc > 2 && std::string(argv[2]) == "--autostart");
    if (cmd == "transcribe" && argc >= 3) {
        bool wake = false;
        for (int i = 3; i < argc; ++i) if (std::string(argv[i]) == "--wake") wake = true;
        return do_transcribe(argv[2], wake);
    }
    if (cmd == "history") return do_history();
    if ((cmd == "show" || cmd == "replay") && argc >= 3) {
        std::string id = argv[2];
        if (!valid_id(id)) return 2;
        fs::path h = data_dir() / "history";
        if (cmd == "show") {
            std::cout << "You: " << slurp(h / (id + ".q")) << "\n\n" << slurp(h / (id + ".a")) << "\n";
            return 0;
        }
        Assets a = assets();
        if (!ensure_tts(a)) return 1;
        speak(a, slurp(h / (id + ".a")));
        return 0;
    }
    std::cout << "kibavoice: local LLM + STT + TTS with a floating UI\n"
                 "  ui [--background] [--listen] | fetch | doctor | install-desktop [--autostart]\n"
                 "  install-service [--no-start] | uninstall-service   (always-on daemon)\n"
                 "  ask <text> | listen [--no-speak] [--no-notify]\n"
                 "  transcribe <wav> [--wake] | say <text> | history | show <id> | replay <id>\n";
    return cmd == "help" ? 0 : 2;
}
