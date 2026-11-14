// Always-on microphone engine. One arecord process stays open while the UI runs.
//   Wake    : cut the stream into short speech segments (for "Hey Kiba" checks)
//   Command : capture one utterance, then pause (the user's actual request)
//   Paused  : keep the device open but throw audio away (while Kiba thinks/speaks)
//   Off     : recorder not running
// Audio never leaves the machine and nothing is kept except the current segment.
#pragma once
#include <QByteArray>
#include <QObject>
#include <QProcess>
#include <deque>
#include <vector>

class MicCapture : public QObject {
    Q_OBJECT
public:
    enum Mode { Off, Wake, Command, Paused };
    explicit MicCapture(QObject* parent = nullptr);
    ~MicCapture() override;

    void setDir(const QString& dir) { dir_ = dir; }
    void setMode(Mode m);
    Mode mode() const { return mode_; }
    void finishNow();   // Command mode: send what we have now (or give up if silent)

signals:
    // rms in 0..1 (perceptual), low/mid/high are spectral weights summing to ~1
    void level(float rms, float low, float mid, float high);
    void utteranceReady(const QString& wavPath, bool command);
    void nothingHeard();
    void error(const QString& message);

private:
    bool ensureProcess();
    void stopProcess();
    void onReadyRead();
    void processFrame(const int16_t* f);
    void finishUtterance();
    void resetAnalysis();

    QProcess* proc_ = nullptr;
    QString dir_;
    QByteArray buf_;
    Mode mode_ = Off;
    std::vector<int16_t> pcm_;
    std::deque<std::vector<int16_t>> pre_;   // pre-roll so the first syllable isn't clipped
    bool started_ = false, calibrated_ = false;
    int calib_ = 0, loud_ = 0, quiet_ = 0, idle_ = 0, len_ = 0, seq_ = 0;
    double noise_ = 0, thr_ = 600;
};
