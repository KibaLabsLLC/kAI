#include "audio.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QFile>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_map>

namespace {
constexpr int kFrame = 320;            // 20 ms at 16 kHz
constexpr int kPreRoll = 10;           // 200 ms
constexpr int kMinWake = 18;           // ignore blips shorter than ~0.36 s
constexpr int kWakeEnd = 30;           // 0.6 s of quiet ends a wake segment
constexpr int kWakeMax = 400;          // 8 s
constexpr int kCmdEnd = 50;            // 1 s of quiet ends a command
constexpr int kCmdMax = 750;           // 15 s
constexpr int kCmdNoSpeech = 300;      // 6 s

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

// ---- Mic hand-off ---------------------------------------------------------------------------
// While we listen, we poll the sound server for *other* apps that are actively recording. When one
// shows up we close our recorder (releasing the mic); once it has been gone for a couple of polls
// we reopen it and re-calibrate. State lives here so audio.h does not need to change.
struct Guard {
    QTimer* timer = nullptr;
    QProcess* probe = nullptr;
    std::function<void(const QString&)> apply;   // feed it the name of the other recorder ("" = none)
    bool yielded = false;     // recorder closed on purpose (or because the device was busy)
    bool noPactl = false;     // no pactl -> only the "device busy, retry" path works
    int clear = 0;            // consecutive polls with no other recorder
    int quickFails = 0;       // recorder died right after start (device busy) in a row
    qint64 startedMs = 0;
    QString who;
};
std::unordered_map<const MicCapture*, Guard> g_guards;

qint64 ppidOf(qint64 pid) {
    QFile f(QString("/proc/%1/stat").arg(pid));
    if (!f.open(QIODevice::ReadOnly)) return 0;
    const QByteArray s = f.readAll();
    const int rp = s.lastIndexOf(')');                     // comm can contain spaces and parens
    if (rp < 0) return 0;
    const QList<QByteArray> t = s.mid(rp + 2).split(' ');   // state ppid ...
    return t.size() > 1 ? t[1].toLongLong() : 0;
}
bool descendantOf(qint64 pid, qint64 root) {
    for (int i = 0; pid > 1 && i < 8; ++i) { if (pid == root) return true; pid = ppidOf(pid); }
    return false;
}

// Name of the first app other than us that is actively recording, or "".
QString otherRecorder(const QString& text, qint64 ourRecorderPid) {
    const qint64 self = QCoreApplication::applicationPid();
    const QStringList blocks = text.split(QRegularExpression("\n(?=Source Output #)"));
    for (const QString& b : blocks) {
        if (!b.contains("Source Output #")) continue;
        if (b.contains(QRegularExpression("Corked:\\s*yes"))) continue;       // paused stream, not using the mic
        auto prop = [&](const QString& key) {
            const auto m = QRegularExpression("\\b" + QRegularExpression::escape(key) + " = \"([^\"]*)\"").match(b);
            return m.hasMatch() ? m.captured(1) : QString();
        };
        if (prop("stream.capture.sink") == "true") continue;                   // listening to a speaker monitor
        if (prop("media.name").contains("peak detect", Qt::CaseInsensitive)) continue;   // volume meters
        const qint64 pid = prop("application.process.id").toLongLong();
        const QString bin = prop("application.process.binary");
        if (pid == self || (pid > 0 && ourRecorderPid > 0 && descendantOf(pid, ourRecorderPid))) continue;   // that's us
        if (pid <= 0 && bin == "arecord") continue;
        const QString name = prop("application.name");
        return !name.isEmpty() ? name : (!bin.isEmpty() ? bin : QStringLiteral("another app"));
    }
    return QString();
}
}  // namespace

MicCapture::MicCapture(QObject* parent) : QObject(parent) {
    Guard& g = g_guards[this];
    g.noPactl = QStandardPaths::findExecutable("pactl").isEmpty();
    g.timer = new QTimer(this);
    g.timer->setInterval(1000);

    g.apply = [this](const QString& who) {
        Guard& G = g_guards[this];
        if (mode_ == Off) return;
        if (!G.yielded) {
            if (who.isEmpty() || !proc_) return;
            qInfo().noquote() << "mic: releasing for" << who;
            G.yielded = true; G.clear = 0; G.who = who;
            stopProcess(); resetAnalysis();
            if (mode_ == Command) { mode_ = Paused; emit nothingHeard(); }   // don't leave the window waiting
            return;
        }
        if (!who.isEmpty()) { G.clear = 0; G.who = who; return; }
        const int need = std::min(2 + G.quickFails, 6);   // debounce; back off while the device keeps refusing us
        if (++G.clear < need) return;
        qInfo() << "mic: reclaiming";
        G.yielded = false; G.clear = 0;
        calibrated_ = false; calib_ = 0; noise_ = 0;
        if (!ensureProcess()) { mode_ = Off; G.timer->stop(); }
    };

    connect(g.timer, &QTimer::timeout, this, [this] {
        Guard& G = g_guards[this];
        if (mode_ == Off) { G.timer->stop(); return; }
        if (G.probe) return;                             // previous poll still running
        if (G.noPactl) { G.apply(QString()); return; }   // can only retry a busy device
        auto* p = new QProcess(this);
        G.probe = p;
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("LC_ALL", "C");                       // "Corked: yes" must not be translated
        p->setProcessEnvironment(env);
        connect(p, &QProcess::errorOccurred, this, [this, p](QProcess::ProcessError e) {
            if (e != QProcess::FailedToStart) return;
            Guard& G2 = g_guards[this];
            G2.noPactl = true; G2.probe = nullptr; p->deleteLater();
        });
        connect(p, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
                [this, p](int code, QProcess::ExitStatus st) {
            Guard& G2 = g_guards[this];
            G2.probe = nullptr;
            const QString out = QString::fromUtf8(p->readAllStandardOutput());
            p->deleteLater();
            if (code != 0 || st != QProcess::NormalExit) { G2.apply(QString()); return; }   // no sound server: nothing to detect
            G2.apply(otherRecorder(out, proc_ ? proc_->processId() : 0));
        });
        QTimer::singleShot(2000, p, [p] { if (p->state() != QProcess::NotRunning) p->kill(); });
        p->start("pactl", {"list", "source-outputs"});
    });
}

MicCapture::~MicCapture() {
    stopProcess();
    auto it = g_guards.find(this);
    if (it != g_guards.end()) {
        if (it->second.probe) { it->second.probe->disconnect(this); it->second.probe->kill(); }
        g_guards.erase(it);
    }
}

void MicCapture::resetAnalysis() {
    started_ = false; loud_ = quiet_ = idle_ = len_ = 0;
    pcm_.clear(); pre_.clear();
}

void MicCapture::setMode(Mode m) {
    if (m == mode_) return;
    mode_ = m;
    resetAnalysis();
    Guard& G = g_guards[this];
    if (m == Off) {
        G.yielded = false; G.clear = 0; G.quickFails = 0; G.timer->stop();
        stopProcess();
        return;
    }
    if (!ensureProcess()) { mode_ = Off; return; }
    if (!G.timer->isActive()) G.timer->start();
}

bool MicCapture::ensureProcess() {
    if (proc_) return true;
    Guard& G = g_guards[this];
    if (G.yielded) return true;        // another app has the mic; the poller reopens us when it is free
    // Same contract as the CLI: KIBAVOICE_REC_CMD overrides the recorder (env/constant only).
    QString cmd = qEnvironmentVariable("KIBAVOICE_REC_CMD");
    if (cmd.isEmpty()) {
        if (QStandardPaths::findExecutable("arecord").isEmpty()) {
            emit error("arecord not found (install alsa-utils)");
            return false;
        }
        cmd = "exec arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null";   // exec: killing the shell must kill arecord itself
    }
    buf_.clear();
    proc_ = new QProcess(this);
    connect(proc_, &QProcess::readyReadStandardOutput, this, &MicCapture::onReadyRead);
    connect(proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) { stopProcess(); mode_ = Off; emit error("could not start the recorder"); }
    });
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus) {
        Guard& G2 = g_guards[this];
        stopProcess();
        if (mode_ == Off) return;
        if (code == 126 || code == 127) {   // recorder missing / not executable: retrying cannot help
            mode_ = Off; G2.yielded = false; G2.timer->stop();
            emit error("microphone stopped");
            return;
        }
        // Anything else (device grabbed by another app, source suspended, hw busy at open): keep
        // wanting the mic and retry with a growing delay for as long as we are not Off.
        if (++G2.quickFails == 1) qInfo() << "mic: recorder stopped (device taken?), will keep retrying";
        G2.yielded = true; G2.clear = 0; G2.who = "another app";
        resetAnalysis();
        if (mode_ == Command) { mode_ = Paused; emit nothingHeard(); }
        G2.timer->start();
    });
    G.startedMs = QDateTime::currentMSecsSinceEpoch();
    proc_->start("/bin/sh", {"-c", cmd});
    return true;
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
    g_guards[this].quickFails = 0;     // data is flowing: the device opened fine
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

    if (mode_ == Command) {   // visualizer feed only while the user is addressing Kiba
        const double lo = goertzel(f, kFrame, 250) + goertzel(f, kFrame, 450);
        const double mid = goertzel(f, kFrame, 900) + goertzel(f, kFrame, 1500);
        const double hi = goertzel(f, kFrame, 2600) + goertzel(f, kFrame, 3800);
        const double sum = lo + mid + hi + 1e-9;
        emit level(float(std::pow(std::min(1.0, rms / 5000.0), 0.6)), float(lo / sum), float(mid / sum), float(hi / sum));
    }

    // noise floor: calibrate once, then keep tracking it while Wake mode hears nothing
    if (!calibrated_) {
        noise_ += rms / 15;
        if (++calib_ == 15) { thr_ = std::max(500.0, noise_ * 3.0); calibrated_ = true; }
        return;
    }

    std::vector<int16_t> frame(f, f + kFrame);
    if (!started_) {
        pre_.push_back(std::move(frame));
        if (int(pre_.size()) > kPreRoll) pre_.pop_front();
        loud_ = rms > thr_ ? loud_ + 1 : 0;
        if (mode_ == Wake && loud_ == 0) {
            noise_ = 0.98 * noise_ + 0.02 * rms;
            thr_ = std::max(500.0, noise_ * 3.0);
        }
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
    quiet_ = rms < thr_ ? quiet_ + 1 : 0;
    const bool wake = (mode_ == Wake);
    if (quiet_ >= (wake ? kWakeEnd : kCmdEnd) || len_ >= (wake ? kWakeMax : kCmdMax)) finishUtterance();
}

void MicCapture::finishUtterance() {
    const bool command = (mode_ == Command);
    const int frames = len_;
    std::vector<int16_t> pcm = std::move(pcm_);
    resetAnalysis();
    if (!command && frames < kMinWake) return;
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
