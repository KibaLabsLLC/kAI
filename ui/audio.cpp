#include "audio.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
constexpr int kFrame = 320;            // 20 ms at 16 kHz
constexpr int kPreRoll = 10;           // 200 ms
constexpr int kMinWake = 18;           // a wake segment needs ~0.36 s of sound (pre-roll included); clicks and coughs are shorter
constexpr int kWakeEnd = 30;           // 0.6 s of quiet ends a wake segment
constexpr int kWakeMax = 300;          // 6 s: "speech" that never pauses is the room, not a person
constexpr int kCmdEnd = 35;            // 0.7 s of quiet ends a command
constexpr int kCmdMax = 750;           // 15 s
constexpr int kCmdNoSpeech = 300;      // 6 s

bool dbg() { static const bool d = qEnvironmentVariableIsSet("KIBAVOICE_DEBUG"); return d; }
// Lowest threshold the detector will use: 0 by default, so even a very quiet mic works (the room noise level still
// sets it: 3x the noise). Raise it in a noisy room:
//   KIBAVOICE_MIC_FLOOR=150
double thrFloor() { static const double f = std::max(0.0, qEnvironmentVariable("KIBAVOICE_MIC_FLOOR", "0").toDouble()); return f; }

struct Recorder { const char* prog; QStringList args; };
const std::vector<Recorder>& recorders() {   // tried in order; if one dies right away the next one is used
    static const std::vector<Recorder> all = {
        {"arecord", {"-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"}},
        {"parecord", {"--raw", "--rate=16000", "--channels=1", "--format=s16le"}},   // PipeWire/Pulse systems without ALSA's default device
    };
    return all;
}


// ---- who else is recording? --------------------------------------------------------------------------------------
// Kiba lets go of the mic while another app is recording and takes it back when that app stops
// (set KIBAVOICE_MIC_YIELD=0 to turn this off).
bool yieldEnabled() { static const bool on = qEnvironmentVariable("KIBAVOICE_MIC_YIELD", "1") != "0"; return on; }

// True if `pid` is `root` or one of its descendants: the recorder, whatever shell started it, anything of ours.
bool descendsFrom(qint64 pid, qint64 root) {
    for (int i = 0; i < 10 && pid > 1; ++i) {
        if (pid == root) return true;
        QFile f(QString("/proc/%1/stat").arg(pid));
        if (!f.open(QIODevice::ReadOnly)) return false;
        const QByteArray s = f.readAll();
        const int rp = s.lastIndexOf(')');
        if (rp < 0) return false;
        const QList<QByteArray> parts = s.mid(rp + 2).split(' ');   // "S <ppid> ..." after the command name
        if (parts.size() < 2) return false;
        pid = parts[1].toLongLong();
    }
    return pid == root;
}

QString propValue(const QString& line, const char* key) {   // `key = "value"`
    const QString k = QString::fromLatin1(key) + " = ";
    if (!line.startsWith(k)) return QString();
    QString v = line.mid(k.size()).trimmed();
    if (v.size() >= 2 && v.startsWith('"') && v.endsWith('"')) v = v.mid(1, v.size() - 2);
    return v;
}

// PipeWire/PulseAudio: output of "pactl list short sources", a marker line, then "pactl list source-outputs" (LC_ALL=C).
// Returns the name of the first app that is really recording, or "". Ignored: our own streams, paused (corked) streams,
// streams on a ".monitor" source (screen/system audio) and volume-meter "peak detect" streams.
QString busyByPactl(const QString& out, qint64 root) {
    static const QString kSplit = "@@SPLIT@@";
    const int sep = out.indexOf(kSplit);
    const QString srcs = sep < 0 ? QString() : out.left(sep), outs = sep < 0 ? out : out.mid(sep + kSplit.size());
    QSet<QString> monitors;
    for (const QString& l : srcs.split('\n')) {
        const QStringList c = l.split('\t');
        if (c.size() >= 2 && c[1].trimmed().endsWith(".monitor")) monitors.insert(c[0].trimmed());
    }
    const QStringList blocks = outs.split("Source Output #");
    for (int i = 1; i < blocks.size(); ++i) {
        QString src, name, bin, media, pid; bool corked = false;
        for (QString l : blocks[i].split('\n')) {
            l = l.trimmed();
            if (l.startsWith("Corked:")) corked = l.mid(7).trimmed() == "yes";
            else if (l.startsWith("Source:")) src = l.mid(7).trimmed();
            else if (const QString v = propValue(l, "application.name"); !v.isEmpty()) name = v;
            else if (const QString v2 = propValue(l, "application.process.binary"); !v2.isEmpty()) bin = v2;
            else if (const QString v3 = propValue(l, "application.process.id"); !v3.isEmpty()) pid = v3;
            else if (const QString v4 = propValue(l, "media.name"); !v4.isEmpty()) media = v4;
        }
        if (corked || monitors.contains(src) || media.contains("peak detect", Qt::CaseInsensitive)) continue;
        bool ok = false; const qint64 p = pid.toLongLong(&ok);
        if (ok && p > 0 && descendsFrom(p, root)) continue;   // that's us
        return !name.isEmpty() ? name : !bin.isEmpty() ? bin : QString("another app");
    }
    return QString();
}

// Plain ALSA (no sound server): a capture substream that is RUNNING and owned by someone who isn't us. Sound servers
// are skipped, since they hold the device on behalf of whichever app is recording (pactl covers that case).
QString busyByAlsa(qint64 root, const QString& base = "/proc/asound") {
    static const QSet<QString> servers{"pipewire", "pipewire-pulse", "pulseaudio", "wireplumber", "jackd"};
    const QDir b(base);
    for (const QString& card : b.entryList({"card*"}, QDir::Dirs | QDir::NoDotAndDotDot)) {
        const QDir cd(b.filePath(card));
        for (const QString& pcm : cd.entryList({"pcm*c"}, QDir::Dirs | QDir::NoDotAndDotDot)) {   // "c" = capture
            const QDir pd(cd.filePath(pcm));
            for (const QString& sub : pd.entryList({"sub*"}, QDir::Dirs | QDir::NoDotAndDotDot)) {
                QFile f(pd.filePath(sub) + "/status");
                if (!f.open(QIODevice::ReadOnly)) continue;
                const QString s = QString::fromLatin1(f.readAll());
                if (!s.contains("RUNNING")) continue;
                const QRegularExpressionMatch m = QRegularExpression("owner_pid\\s*:\\s*(\\d+)").match(s);
                if (!m.hasMatch()) continue;
                const qint64 pid = m.captured(1).toLongLong();
                if (descendsFrom(pid, root)) continue;
                QFile c(QString("/proc/%1/comm").arg(pid));
                const QString comm = c.open(QIODevice::ReadOnly) ? QString::fromLocal8Bit(c.readAll()).trimmed() : QString();
                if (servers.contains(comm)) continue;
                return comm.isEmpty() ? QString("another app") : comm;
            }
        }
    }
    return QString();
}

double goertzel(const int16_t* x, int n, double hz) {
    const double k = 2.0 * std::cos(2.0 * M_PI * hz / 16000.0);
    double s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < n; ++i) { s0 = x[i] + k * s1 - s2; s2 = s1; s1 = s0; }
    return std::sqrt(std::max(0.0, s1 * s1 + s2 * s2 - k * s1 * s2)) / n;
}

bool writeWav(const QString& path, const std::vector<int16_t>& pcm) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    const uint32_t rate = 16000, data = uint32_t(pcm.size() * 2), riff = 36 + data, fmt = 16, br = rate * 2;
    const uint16_t pcmfmt = 1, ch = 1, align = 2, bits = 16;
    f.write("RIFF", 4); f.write((const char*)&riff, 4); f.write("WAVEfmt ", 8);
    f.write((const char*)&fmt, 4); f.write((const char*)&pcmfmt, 2); f.write((const char*)&ch, 2);
    f.write((const char*)&rate, 4); f.write((const char*)&br, 4); f.write((const char*)&align, 2);
    f.write((const char*)&bits, 2); f.write("data", 4); f.write((const char*)&data, 4);
    f.write((const char*)pcm.data(), data);
    return true;
}
}  // namespace

MicCapture::MicCapture(QObject* parent) : QObject(parent) {
    watch_ = new QTimer(this);
    watch_->setInterval(1000);
    connect(watch_, &QTimer::timeout, this, &MicCapture::watchTick);
}
MicCapture::~MicCapture() {
    if (probe_) { probe_->disconnect(this); probe_->kill(); probe_->waitForFinished(200); }
    stopProcess();
}

void MicCapture::resetAnalysis() {
    started_ = false; loud_ = quiet_ = idle_ = len_ = 0;
    pcm_.clear(); pre_.clear();
}

void MicCapture::setMode(Mode m) {
    if (m == mode_) return;
    const Mode prev = mode_;
    mode_ = m;
    resetAnalysis();
    if (m == Off) {
        stopProcess(); watch_->stop();
        yielded_ = false; busyTicks_ = freeTicks_ = 0; notifyYield(false, QString());
        return;
    }
    if (prev == Off) fails_ = 0;
    if (m == Command && yielded_) { yielded_ = false; notifyYield(false, QString()); }   // the user just asked for Kiba: take the mic back
    if (yieldEnabled() && !watch_->isActive()) watch_->start();
    if (!ensureProcess()) mode_ = Off;
}

// ---- yielding the mic to other apps -----------------------------------------------------------------------------
void MicCapture::notifyYield(bool y, const QString& who) {   // tell the UI only when the state it knows about changes
    if (uiYielded_ == y) return;
    uiYielded_ = y;
    emit micYielded(y, who);
}

void MicCapture::yieldMic(const QString& who) {
    yielded_ = true; freeTicks_ = 0;
    stopProcess(); resetAnalysis();   // really let go of the device
    if (dbg()) std::fprintf(stderr, "mic: %s is recording, so Kiba is letting go of the mic\n", qPrintable(who));
    notifyYield(true, who);
}

void MicCapture::reclaimMic() {
    yielded_ = false; busyTicks_ = freeTicks_ = 0;
    fails_ = 3;   // if the device is somehow still busy, one more failure hands it back again instead of retrying noisily
    resetAnalysis();
    if (dbg()) std::fprintf(stderr, "mic: nobody else is recording, Kiba is taking the mic back\n");
    if (!ensureProcess()) { mode_ = Off; return; }
    QTimer::singleShot(3000, this, [this] { if (!yielded_ && proc_) notifyYield(false, QString()); });   // tell the UI once it has held
}

void MicCapture::watchTick() {
    if (mode_ == Off || probe_) return;
    if (mode_ == Command) { busyTicks_ = freeTicks_ = 0; return; }   // an explicit request from the user always wins
    const qint64 self = QCoreApplication::applicationPid();
    if (QStandardPaths::findExecutable("pactl").isEmpty()) { applyProbe(busyByAlsa(self)); return; }
    probe_ = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("LC_ALL", "C");   // stable field names
    probe_->setProcessEnvironment(env);
    connect(probe_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, self](int code, QProcess::ExitStatus) {
        const QString out = QString::fromUtf8(probe_->readAllStandardOutput());
        probe_->deleteLater(); probe_ = nullptr;
        applyProbe(code == 0 ? busyByPactl(out, self) : busyByAlsa(self));   // no sound server running: fall back to plain ALSA
    });
    connect(probe_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart && probe_) { probe_->deleteLater(); probe_ = nullptr; }
    });
    probe_->start("/bin/sh", {"-c", "pactl list short sources; echo @@SPLIT@@; pactl list source-outputs"});
    QTimer::singleShot(3000, probe_, [p = probe_] { if (p->state() != QProcess::NotRunning) p->kill(); });   // a hung sound server must not stall us
}

// Two readings in a row before acting either way, so a stream that blips on and off can't make Kiba flap.
void MicCapture::applyProbe(const QString& who) {
    if (mode_ == Off || mode_ == Command) return;
    if (!who.isEmpty()) {
        freeTicks_ = 0;
        if (++busyTicks_ >= 2 && !yielded_) yieldMic(who);
    } else {
        busyTicks_ = 0;
        if (yielded_ && ++freeTicks_ >= 2) reclaimMic();
    }
}

bool MicCapture::ensureProcess() {
    if (proc_ || yielded_) return true;   // (while yielded there is deliberately no recorder)
    QString prog; QStringList args;
    // Same contract as the CLI: KIBAVOICE_REC_CMD overrides the recorder (env/constant only).
    const QString cmd = qEnvironmentVariable("KIBAVOICE_REC_CMD");
    if (!cmd.isEmpty()) {
        prog = "/bin/sh"; args = {"-c", cmd};
    } else {
        std::vector<const Recorder*> have;
        for (const auto& r : recorders()) if (!QStandardPaths::findExecutable(r.prog).isEmpty()) have.push_back(&r);
        if (have.empty()) { emit error("no recorder found (install alsa-utils)"); return false; }
        const Recorder& r = *have[size_t(recIdx_) % have.size()];
        prog = r.prog; args = r.args;   // started directly, not through a shell, so stopping it really stops it
    }
    buf_.clear(); lastErr_.clear();
    proc_ = new QProcess(this);
    connect(proc_, &QProcess::readyReadStandardOutput, this, &MicCapture::onReadyRead);
    connect(proc_, &QProcess::readyReadStandardError, this, [this] {   // keep the recorder's own explanation instead of discarding it
        if (!proc_) return;
        const QString s = QString::fromLocal8Bit(proc_->readAllStandardError()).trimmed();
        if (!s.isEmpty()) lastErr_ = s.section('\n', -1);
    });
    connect(proc_, &QProcess::errorOccurred, this, [this, prog](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) { stopProcess(); recorderGone("could not start " + prog); }
    });
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
        if (proc_) {
            const QString s = QString::fromLocal8Bit(proc_->readAllStandardError()).trimmed();
            if (!s.isEmpty()) lastErr_ = s.section('\n', -1);
        }
        const QString why = lastErr_;
        stopProcess();
        recorderGone(why);
    });
    startMs_ = QDateTime::currentMSecsSinceEpoch();
    proc_->start(prog, args);
    return true;
}

// The recorder exited (device busy, no default device, server restart...). Retry quietly, switching recorder each
// time, so a hiccup never reaches the UI; only after repeated quick failures is the mic reported as stopped.
void MicCapture::recorderGone(const QString& why) {
    const Mode want = mode_;
    if (want == Off) return;
    if (QDateTime::currentMSecsSinceEpoch() - startMs_ > 10000) fails_ = 0;   // it had been running fine
    ++fails_; ++recIdx_;
    if (dbg()) std::fprintf(stderr, "kibavoice-ui: recorder stopped (%s), attempt %d\n", qPrintable(why.isEmpty() ? QString("no message") : why), fails_);
    if (fails_ > 3) {
        fails_ = 0;
        const bool busy = why.contains("busy", Qt::CaseInsensitive) || why.contains("in use", Qt::CaseInsensitive);
        if (busy && yieldEnabled()) {   // someone else has the device exclusively: wait for them instead of giving up
            if (!yielded_) { yielded_ = true; busyTicks_ = freeTicks_ = 0; notifyYield(true, QString("another app")); }
            return;
        }
        mode_ = Off;
        emit error(why.isEmpty() ? QString("microphone stopped") : "microphone stopped: " + why);
        return;
    }
    QTimer::singleShot(fails_ * 500, this, [this, want] {
        if (!proc_ && !yielded_ && mode_ == want && !ensureProcess()) mode_ = Off;
    });
}

void MicCapture::stopProcess() {
    if (!proc_) return;
    proc_->disconnect(this);
    proc_->terminate();
    if (!proc_->waitForFinished(300)) proc_->kill();
    proc_->deleteLater();
    proc_ = nullptr;
    buf_.clear();
}

void MicCapture::onReadyRead() {
    if (!proc_) return;
    buf_ += proc_->readAllStandardOutput();
    const int bytes = kFrame * 2;
    while (proc_ && buf_.size() >= bytes) {
        int16_t f[kFrame];
        std::memcpy(f, buf_.constData(), bytes);
        buf_.remove(0, bytes);
        processFrame(f);
    }
}

void MicCapture::processFrame(const int16_t* f) {
    if (mode_ != Wake && mode_ != Command) return;   // Paused/Off: discard

    double e = 0;
    for (int i = 0; i < kFrame; ++i) e += double(f[i]) * f[i];
    const double rms = std::sqrt(e / kFrame);

    if (dbg()) {   // once a second: loudest frame heard vs the threshold it has to beat
        static double peak = 0; static int tick = 0;
        peak = std::max(peak, rms);
        if (++tick % 50 == 0) { std::fprintf(stderr, "mic %-4s peak=%5.0f noise=%5.0f threshold=%5.0f%s\n", mode_ == Wake ? "wake" : "cmd", peak, noise_, thr_, started_ ? "  (hearing sound)" : ""); peak = 0; }
    }

    if (mode_ == Command) {   // visualizer feed only while the user is addressing Kiba
        const double lo = goertzel(f, kFrame, 250) + goertzel(f, kFrame, 450);
        const double mid = goertzel(f, kFrame, 900) + goertzel(f, kFrame, 1500);
        const double hi = goertzel(f, kFrame, 2600) + goertzel(f, kFrame, 3800);
        const double sum = lo + mid + hi + 1e-9;
        emit level(float(std::pow(std::min(1.0, rms / 5000.0), 0.6)), float(lo / sum), float(mid / sum), float(hi / sum));
    }

    // Calibrate on the first ~0.4 s (skipping the recorder's start-up pop), then keep tracking the noise floor.
    if (!calibrated_) {
        if (++calib_ > 5) noise_ += rms / 15;
        if (calib_ == 20) { thr_ = std::max(thrFloor(), noise_ * 3.0); calibrated_ = true; if (dbg()) std::fprintf(stderr, "mic calibrated: noise=%.0f threshold=%.0f\n", noise_, thr_); }
        return;
    }

    std::vector<int16_t> frame(f, f + kFrame);
    if (!started_) {
        pre_.push_back(std::move(frame));
        if (int(pre_.size()) > kPreRoll) pre_.pop_front();
        loud_ = rms > thr_ ? loud_ + 1 : 0;
        // Follow the noise floor down quickly and up slowly, whatever the frame. (It used to adapt only on frames
        // already below the threshold, so a room that got louder could never raise it: everything counted as speech.)
        noise_ += (rms < noise_ ? 0.1 : 0.005) * (rms - noise_);
        thr_ = std::max(thrFloor(), noise_ * 3.0);
        if (loud_ >= 3) {
            started_ = true; quiet_ = 0;
            for (auto& p : pre_) pcm_.insert(pcm_.end(), p.begin(), p.end());
            len_ = int(pre_.size());
            pre_.clear();
        } else if (mode_ == Command && ++idle_ > kCmdNoSpeech) {
            mode_ = Paused; resetAnalysis();
            emit nothingHeard();
        }
        return;
    }

    pcm_.insert(pcm_.end(), frame.begin(), frame.end());
    ++len_;
    quiet_ = rms <= thr_ ? quiet_ + 1 : 0;   // <=, so a threshold of 0 still ends on digital silence
    const bool wake = (mode_ == Wake);
    if (quiet_ >= (wake ? kWakeEnd : kCmdEnd) || len_ >= (wake ? kWakeMax : kCmdMax)) finishUtterance();
}

void MicCapture::finishUtterance() {
    const bool command = (mode_ == Command);
    const int frames = len_, voiced = len_ - quiet_;   // voiced = without the trailing silence that ended it
    std::vector<int16_t> pcm = std::move(pcm_);
    resetAnalysis();
    if (!command) {
        if (voiced < kMinWake) { if (dbg()) std::fprintf(stderr, "mic: ignored a %.2f s blip\n", voiced * 0.02); return; }
        if (frames >= kWakeMax) {   // it never went quiet: treat it as the new background level, not as speech
            double e = 0; for (int16_t s : pcm) e += double(s) * s;
            const double r = std::sqrt(e / double(std::max<size_t>(1, pcm.size())));
            noise_ = std::max(noise_, std::min(r, 2000.0));
            thr_ = std::max(thrFloor(), noise_ * 3.0);
            if (dbg()) std::fprintf(stderr, "mic: %.0f s of constant sound, treating it as background (noise=%.0f threshold=%.0f)\n", frames * 0.02, noise_, thr_);
            return;
        }
        if (dbg()) std::fprintf(stderr, "mic: wake segment %.1f s sent to speech recognition\n", frames * 0.02);
    }
    if (command) {
        mode_ = Paused;   // the window decides what happens next
        emit level(0, 0.34f, 0.33f, 0.33f);
    }
    const QString path = command ? dir_ + "/tmp-command.wav"
                                 : dir_ + "/tmp-wake-" + QString::number(seq_++ % 1000) + ".wav";
    if (!writeWav(path, pcm)) { emit error("could not write the recording"); return; }
    emit utteranceReady(path, command);
}

void MicCapture::finishNow() {
    if (mode_ != Command) return;
    if (started_) finishUtterance();
    else { mode_ = Paused; resetAnalysis(); emit nothingHeard(); }
}
