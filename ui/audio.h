// Microphone capture for the UI: runs arecord, reports live levels for the
// visualizer, and ends the utterance with the same VAD rules as the CLI.
#pragma once
#include <QByteArray>
#include <QObject>
#include <QProcess>
#include <vector>

class MicCapture : public QObject {
    Q_OBJECT
public:
    explicit MicCapture(QObject* parent = nullptr);
    ~MicCapture() override;
    bool active() const { return active_; }
    void start(const QString& wavPath);
    void finishNow();   // user tapped stop: send what we have (or cancel if no speech)
    void cancel();

signals:
    // rms in 0..1 (perceptual), low/mid/high are spectral weights summing to ~1
    void level(float rms, float low, float mid, float high);
    void utteranceReady(const QString& wavPath);
    void nothingHeard();
    void error(const QString& message);

private:
    void onReadyRead();
    void processFrame(const int16_t* f);
    void stopProcess();
    void finish(bool send);

    QProcess* proc_ = nullptr;
    QString wav_;
    QByteArray buf_;
    std::vector<int16_t> pcm_;
    bool active_ = false, started_ = false;
    int n_ = 0, calib_ = 0, loud_ = 0, quiet_ = 0;
    double noise_ = 0, thr_ = 600;
};
