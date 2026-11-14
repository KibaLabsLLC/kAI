#include "audio.h"

#include <QFile>
#include <QStandardPaths>
#include <cmath>
#include <cstring>

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
}  // namespace

MicCapture::MicCapture(QObject* parent) : QObject(parent) {}
MicCapture::~MicCapture() { stopProcess(); }

void MicCapture::resetAnalysis() {
    started_ = false; loud_ = quiet_ = idle_ = len_ = 0;
    pcm_.clear(); pre_.clear();
}

void MicCapture::setMode(Mode m) {
    if (m == mode_) return;
    mode_ = m;
    resetAnalysis();
    if (m == Off) { stopProcess(); return; }
    if (!ensureProcess()) mode_ = Off;
}

bool MicCapture::ensureProcess() {
    if (proc_) return true;
    // Same contract as the CLI: KIBAVOICE_REC_CMD overrides the recorder (env/constant only).
    QString cmd = qEnvironmentVariable("KIBAVOICE_REC_CMD");
    if (cmd.isEmpty()) {
        if (QStandardPaths::findExecutable("arecord").isEmpty()) {
            emit error("arecord not found (install alsa-utils)");
            return false;
        }
        cmd = "arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null";
    }
    buf_.clear();
    proc_ = new QProcess(this);
    connect(proc_, &QProcess::readyReadStandardOutput, this, &MicCapture::onReadyRead);
    connect(proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) { stopProcess(); mode_ = Off; emit error("could not start the recorder"); }
    });
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
        stopProcess(); mode_ = Off; emit error("microphone stopped");
    });
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
