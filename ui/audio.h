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
#include <QString>
#include <QTimer>
#include <cstdint>
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
    // Emitted by notifyYield() only when the yield state the UI knows about changes.
    void micYielded(bool yielded, const QString& who);

private slots:
    void watchTick();   // QTimer::timeout target

private:
    bool ensureProcess();
    void stopProcess();
    void onReadyRead();
    void processFrame(const int16_t* f);
    void finishUtterance();
    void resetAnalysis();

    // Mic yielding: step aside when another app wants the mic, reclaim it when free
    void notifyYield(bool y, const QString& who);
    void yieldMic(const QString& who);
    void reclaimMic();
    void applyProbe(const QString& who);
    void recorderGone(const QString& why);

    QProcess* proc_ = nullptr;
    QProcess* probe_ = nullptr;   // checks whether someone else is using the mic
    QTimer* watch_ = nullptr;     // polls for mic contention
    QString dir_;
    QString lastErr_;             // recorder's last stderr line
    QByteArray buf_;
    Mode mode_ = Off;
    std::vector<int16_t> pcm_;
    std::deque<std::vector<int16_t>> pre_;   // pre-roll so the first syllable isn't clipped
    bool started_ = false, calibrated_ = false;
    bool yielded_ = false;        // true while we've released the mic (no recorder running)
    bool uiYielded_ = false;      // last yield state the UI was told about
    int calib_ = 0, loud_ = 0, quiet_ = 0, idle_ = 0, len_ = 0, seq_ = 0;
    int busyTicks_ = 0, freeTicks_ = 0, fails_ = 0, recIdx_ = 0;
    qint64 startMs_ = 0;
    double noise_ = 0, thr_ = 600;
};
