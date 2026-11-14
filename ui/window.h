#pragma once
#include <QLabel>
#include <QTimer>
#include <QWidget>

#include "audio.h"
#include "backend.h"
#include "widgets.h"

// Voice-only floating window: the mascot, a status line, and the wave.
// Say "Hey Kiba" (optionally followed by the request) and it answers out loud.
class FloatingWindow : public QWidget {
    Q_OBJECT
public:
    enum State { Setup, Idle, Listening, Transcribing, Thinking, Speaking };
    explicit FloatingWindow(bool background, QWidget* parent = nullptr);
    // pin=true: an explicit launch (menu, tray); in background mode the window then stays up
    // until the user closes it, instead of tucking itself away after the answer.
    void present(bool listenNow, bool pin = false);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    void closeEvent(QCloseEvent*) override;

private:
    enum Pending { None, WakeCheck, Command };

    void setState(State s, const QString& status = {}, bool isError = false);
    void showStatus(const QString& text, bool isError);
    void refreshIdleStatus();
    void startCommand();
    void onSegment(const QString& wav, bool command);
    void onTranscribed(const QString& text);
    void onFailed(const QString& message);
    void runWakeCheck(const QString& wav);
    void runHeld();
    QString takeHeld();
    void askQuestion(const QString& q);
    void finishAnswer(const QString& answer);
    void wavePressed();

    State state_ = Setup;
    Pending pending_ = None;
    bool muted_ = false, background_ = false, micBroken_ = false, setupFailed_ = false, pinned_ = false;
    QString held_;   // a segment captured while the wake check was still running
    Backend backend_;
    MicCapture mic_;

    MascotWidget* mascot_;
    QLabel* status_;
    IconButton *closeBtn_, *soundBtn_;
    WaveWidget* wave_;
    QTimer statusReset_, resumeTimer_, hideTimer_, retryTimer_;
};
