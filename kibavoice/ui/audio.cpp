#include "audio.h"

#include <QFile>
#include <QStandardPaths>
#include <cmath>
#include <cstring>

namespace {
constexpr int kFrame = 320;          // 20 ms at 16 kHz
constexpr int kMaxFrames = 750;      // 15 s
constexpr int kNoSpeechFrames = 300; // 6 s
constexpr int kSilenceFrames = 50;   // 1 s

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

void MicCapture::start(const QString& wavPath) {
    if (active_) return;
    // Same contract as the CLI: KIBAVOICE_REC_CMD overrides the recorder (env/constant only).
    QString cmd = qEnvironmentVariable("KIBAVOICE_REC_CMD");
    if (cmd.isEmpty()) {
        if (QStandardPaths::findExecutable("arecord").isEmpty()) {
            emit error("arecord not found (install alsa-utils)");
            return;
        }
        cmd = "arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null";
    }
    wav_ = wavPath;
    buf_.clear(); pcm_.clear();
    started_ = false; n_ = calib_ = loud_ = quiet_ = 0; noise_ = 0; thr_ = 600;
    active_ = true;
    proc_ = new QProcess(this);
    proc_->setProcessChannelMode(QProcess::SeparateChannels);
    connect(proc_, &QProcess::readyReadStandardOutput, this, &MicCapture::onReadyRead);
    connect(proc_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart && active_) { active_ = false; emit error("could not start the recorder"); }
    });
    connect(proc_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
        if (active_) finish(started_);   // recorder died early: use what we have
    });
    proc_->start("/bin/sh", {"-c", cmd});
}

void MicCapture::onReadyRead() {
    if (!active_) return;
    buf_ += proc_->readAllStandardOutput();
    const int bytes = kFrame * 2;
    while (active_ && buf_.size() >= bytes) {
        int16_t f[kFrame];
        std::memcpy(f, buf_.constData(), bytes);
        buf_.remove(0, bytes);
        processFrame(f);
    }
}

void MicCapture::processFrame(const int16_t* f) {
    pcm_.insert(pcm_.end(), f, f + kFrame);
    double e = 0;
    for (int i = 0; i < kFrame; ++i) e += double(f[i]) * f[i];
    const double rms = std::sqrt(e / kFrame);

    // visualizer feed
    const double lo = goertzel(f, kFrame, 250) + goertzel(f, kFrame, 450);
    const double mid = goertzel(f, kFrame, 900) + goertzel(f, kFrame, 1500);
    const double hi = goertzel(f, kFrame, 2600) + goertzel(f, kFrame, 3800);
    const double sum = lo + mid + hi + 1e-9;
    const float lvl = float(std::pow(std::min(1.0, rms / 5000.0), 0.6));
    emit level(lvl, float(lo / sum), float(mid / sum), float(hi / sum));

    // end-of-utterance detection (mirrors the CLI)
    ++n_;
    if (calib_ < 15) {
        noise_ += rms / 15;
        if (++calib_ == 15) thr_ = std::max(500.0, noise_ * 3.0);
    } else if (!started_) {
        loud_ = rms > thr_ ? loud_ + 1 : 0;
        if (loud_ >= 3) started_ = true;
        else if (n_ > kNoSpeechFrames) { finish(false); return; }
    } else {
        quiet_ = rms < thr_ ? quiet_ + 1 : 0;
        if (quiet_ >= kSilenceFrames) { finish(true); return; }
    }
    if (n_ >= kMaxFrames) finish(started_);
}

void MicCapture::finishNow() { if (active_) finish(started_); }
void MicCapture::cancel() { if (active_) finish(false); }

void MicCapture::stopProcess() {
    if (!proc_) return;
    proc_->disconnect(this);
    proc_->terminate();
    if (!proc_->waitForFinished(300)) proc_->kill();
    proc_->deleteLater();
    proc_ = nullptr;
}

void MicCapture::finish(bool send) {
    if (!active_) return;
    active_ = false;
    stopProcess();
    emit level(0, 0.34f, 0.33f, 0.33f);
    if (!send) { emit nothingHeard(); return; }
    if (!writeWav(wav_, pcm_)) { emit error("could not write the recording"); return; }
    emit utteranceReady(wav_);
}
