#pragma once
#include <QLabel>
#include <QLineEdit>
#include <QTimer>
#include <QWidget>

#include "audio.h"
#include "backend.h"
#include "widgets.h"

class FloatingWindow : public QWidget {
    Q_OBJECT
public:
    enum State { Idle, Listening, Transcribing, Thinking, Speaking };
    explicit FloatingWindow(QWidget* parent = nullptr);
    void present(bool listen);

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void resizeEvent(QResizeEvent*) override;
    void closeEvent(QCloseEvent*) override;

private:
    void setState(State s, const QString& status = {}, bool isError = false);
    void startListening();
    void micClicked();
    void submitText();
    void askQuestion(const QString& q);
    void finishAnswer(const QString& q, const QString& a);
    void updateMicButton();

    State state_ = Idle;
    bool muted_ = false;
    QString wav_;
    Backend backend_;
    MicCapture mic_;

    MascotWidget* mascot_;
    QLabel* status_;
    ChatView* chat_;
    QLineEdit* input_;
    IconButton *micBtn_, *closeBtn_, *soundBtn_;
    WaveWidget* wave_;
    QTimer statusReset_;
};
