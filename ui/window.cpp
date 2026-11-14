#include "window.h"
#include "theme.h"
#include "wake.h"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QPainter>
#include <QScreen>
#include <QSettings>
#include <QVBoxLayout>
#include <QWindow>

namespace {
constexpr int kMargin = 22;   // room for the drop shadow around the card
constexpr int kRadius = 28;
const char* const kWakeHint = "Say “Hey Kiba”";
void rm(const QString& path) { if (!path.isEmpty()) QFile::remove(path); }
}

FloatingWindow::FloatingWindow(bool background, QWidget* parent) : QWidget(parent), background_(background) {
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowTitle("Kiba");
    resize(320 + 2 * kMargin, 360 + 2 * kMargin);
    setFixedSize(size());
    muted_ = QSettings("kibavoice", "ui").value("muted", false).toBool();

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(kMargin, kMargin, kMargin, kMargin);
    root->setSpacing(0);

    mascot_ = new MascotWidget;
    mascot_->setFixedHeight(190);
    root->addWidget(mascot_);

    status_ = new QLabel;
    status_->setAlignment(Qt::AlignCenter);
    status_->setFont(theme::font(11, QFont::Medium));
    status_->setFixedHeight(28);
    root->addWidget(status_);
    root->addStretch(1);

    wave_ = new WaveWidget;
    wave_->setFixedHeight(120);
    root->addWidget(wave_);

    closeBtn_ = new IconButton(IconButton::Close, 30, this);
    soundBtn_ = new IconButton(muted_ ? IconButton::SoundOff : IconButton::SoundOn, 30, this);
    closeBtn_->raise(); soundBtn_->raise();

    connect(closeBtn_, &IconButton::clicked, this, &QWidget::close);
    connect(soundBtn_, &IconButton::clicked, this, [this] {
        muted_ = !muted_;
        QSettings("kibavoice", "ui").setValue("muted", muted_);
        soundBtn_->setKind(muted_ ? IconButton::SoundOff : IconButton::SoundOn);
        if (muted_ && state_ == Speaking) backend_.stopSpeaking();
    });
    // Manual fallbacks: tap the wave to talk without the wake phrase, tap Kiba to stop speaking.
    connect(wave_, &WaveWidget::clicked, this, &FloatingWindow::wavePressed);
    connect(mascot_, &MascotWidget::clicked, this, [this] {
        if (state_ == Speaking) backend_.stopSpeaking(); else wavePressed();
    });

    mic_.setDir(Backend::dataDir());
    connect(&mic_, &MicCapture::level, wave_, &WaveWidget::setLevel);
    connect(&mic_, &MicCapture::utteranceReady, this, &FloatingWindow::onSegment);
    connect(&mic_, &MicCapture::nothingHeard, this, [this] { setState(Idle, "Didn't catch that", true); });
    connect(&mic_, &MicCapture::error, this, [this](const QString& m) {
        micBroken_ = true;
        pending_ = None;
        setState(Idle);   // with micBroken_ set this leaves the recorder off
        showStatus("Mic unavailable: " + m, true);
        statusReset_.stop();
        if (background_) retryTimer_.start(15000);   // daemon: keep trying, e.g. device busy or unplugged
    });

    connect(&backend_, &Backend::transcribed, this, &FloatingWindow::onTranscribed);
    connect(&backend_, &Backend::answered, this, [this](const QString&, const QString& a) { finishAnswer(a); });
    connect(&backend_, &Backend::failed, this, &FloatingWindow::onFailed);
    connect(&backend_, &Backend::speechFinished, this, [this] { if (state_ == Speaking) setState(Idle); });
    connect(&backend_, &Backend::fetched, this, [this](bool ok) {
        if (ok) { setupFailed_ = false; setState(Idle); return; }
        setupFailed_ = true;
        micBroken_ = true;
        setState(Idle);
        showStatus("Setup failed (offline?). Run: kibavoice fetch", true);
        statusReset_.stop();
        if (background_) retryTimer_.start(60000);   // daemon at boot: the network may simply not be up yet
    });

    statusReset_.setSingleShot(true);
    statusReset_.setInterval(4500);
    connect(&statusReset_, &QTimer::timeout, this, &FloatingWindow::refreshIdleStatus);

    resumeTimer_.setSingleShot(true);   // short pause after speaking so Kiba doesn't hear itself
    resumeTimer_.setInterval(600);
    connect(&resumeTimer_, &QTimer::timeout, this, [this] {
        if (state_ == Idle && !micBroken_) mic_.setMode(MicCapture::Wake);
    });

    hideTimer_.setSingleShot(true);
    hideTimer_.setInterval(3500);
    connect(&hideTimer_, &QTimer::timeout, this, [this] { if (background_ && !pinned_ && state_ == Idle) hide(); });

    retryTimer_.setSingleShot(true);   // daemon self-healing: broken mic / failed first-run download
    connect(&retryTimer_, &QTimer::timeout, this, [this] {
        if (!micBroken_ || state_ != Idle) return;
        micBroken_ = false;
        if (setupFailed_ || !backend_.hasStt() || !backend_.hasLlm()) {
            setupFailed_ = false;
            setState(Setup);
            backend_.fetchModels();
        } else {
            setState(Idle);
        }
    });

    if (!backend_.hasStt() || !backend_.hasLlm()) {
        setState(Setup);
        backend_.fetchModels();
    } else {
        setState(Idle);
    }
}

void FloatingWindow::present(bool listenNow, bool pin) {
    if (pin) { pinned_ = true; hideTimer_.stop(); }
    if (!isVisible()) {
        if (QScreen* s = QGuiApplication::primaryScreen()) {
            const QRect a = s->availableGeometry();
            move(a.center().x() - width() / 2, a.top() + int(a.height() * 0.14));
        }
    }
    show();
    raise();
    activateWindow();
    if (listenNow) startCommand();
}

// ------------------------------------------------------------- state -------
void FloatingWindow::showStatus(const QString& text, bool isError) {
    status_->setText(text);
    status_->setStyleSheet(QString("color:%1; background:transparent;").arg(isError ? "#D6568F" : "#7B82B8"));
}

void FloatingWindow::refreshIdleStatus() {
    if (state_ == Idle && !micBroken_) showStatus(kWakeHint, false);
}

void FloatingWindow::setState(State s, const QString& status, bool isError) {
    state_ = s;
    QString text = status;
    if (text.isEmpty()) {
        switch (s) {
            case Setup: text = "Getting ready… first run downloads models"; break;
            case Idle: text = kWakeHint; break;
            case Listening: text = "Listening…"; break;
            case Transcribing: text = "Transcribing…"; break;
            case Thinking: text = "Thinking…"; break;
            case Speaking: text = "Speaking · tap Kiba to stop"; break;
        }
    }
    showStatus(text, isError);
    if (s == Idle && !status.isEmpty()) statusReset_.start(); else statusReset_.stop();

    mascot_->setActive(s != Idle);
    mascot_->setFrameMs(s == Listening ? 170 : s == Speaking ? 190 : 230);
    wave_->setMode(s == Listening ? WaveWidget::Listening
                   : s == Speaking ? WaveWidget::Speaking
                   : s == Idle ? WaveWidget::Idle : WaveWidget::Thinking);
    if (s != Listening) wave_->setLevel(0, 0.34f, 0.33f, 0.33f);

    // microphone: only "Hey Kiba" detection while idle; paused while Kiba works or talks
    switch (s) {
        case Setup: mic_.setMode(MicCapture::Off); break;
        case Idle: mic_.setMode(micBroken_ ? MicCapture::Off : MicCapture::Paused); resumeTimer_.start(); break;
        case Listening: break;   // startCommand() switches the mic to Command mode
        default: resumeTimer_.stop(); mic_.setMode(micBroken_ ? MicCapture::Off : MicCapture::Paused); break;
    }

    // background mode: pop up when something happens, tuck away a moment after
    if (background_) {
        if (s != Idle) { hideTimer_.stop(); if (!isVisible()) present(false); }
        else if (isVisible()) hideTimer_.start();
    }
}

// ---------------------------------------------------------- voice flow -----
void FloatingWindow::wavePressed() {
    if (micBroken_) { micBroken_ = false; setState(Idle); return; }
    if (state_ == Idle) startCommand();
    else if (state_ == Listening) mic_.finishNow();
}

void FloatingWindow::startCommand() {
    if (state_ == Speaking) backend_.stopSpeaking();
    if (state_ != Idle) return;
    QDir().mkpath(Backend::dataDir());
    rm(takeHeld());
    pending_ = None;
    setState(Listening);
    mic_.setMode(MicCapture::Command);
}

void FloatingWindow::onSegment(const QString& wav, bool command) {
    if (command) {
        if (state_ != Listening) { rm(wav); return; }
        pending_ = Command;
        setState(Transcribing);
        backend_.transcribe(wav, false);
        return;
    }
    if (state_ != Idle) { rm(wav); return; }
    if (!backend_.busy()) runWakeCheck(wav);
    else { rm(held_); held_ = wav; }   // keep only the newest while a check runs
}

void FloatingWindow::runWakeCheck(const QString& wav) {
    pending_ = WakeCheck;
    backend_.transcribe(wav, true);
}

QString FloatingWindow::takeHeld() { QString h = held_; held_.clear(); return h; }

void FloatingWindow::runHeld() {
    const QString h = takeHeld();
    if (!h.isEmpty() && state_ == Idle) runWakeCheck(h); else rm(h);
}

void FloatingWindow::onTranscribed(const QString& text) {
    if (pending_ == WakeCheck) {
        pending_ = None;
        const WakeMatch m = matchWake(text);
        if (!m.matched || state_ != Idle) { runHeld(); return; }
        if (!m.rest.isEmpty()) { rm(takeHeld()); askQuestion(m.rest); return; }  // "Hey Kiba, what's…"
        const QString h = takeHeld();
        if (!h.isEmpty()) {   // user already started their request while we were checking
            pending_ = Command;
            setState(Transcribing);
            backend_.transcribe(h, false);
        } else {
            startCommand();
        }
    } else if (pending_ == Command) {
        pending_ = None;
        askQuestion(text);
    }
}

void FloatingWindow::onFailed(const QString& message) {
    if (pending_ == WakeCheck) { pending_ = None; runHeld(); return; }   // nothing intelligible: stay quiet
    pending_ = None;
    setState(Idle, message, true);
}

void FloatingWindow::askQuestion(const QString& q) {
    setState(Thinking, backend_.hasLlm() ? "Thinking…" : "First run: downloading language model…");
    backend_.ask(q);
}

void FloatingWindow::finishAnswer(const QString& answer) {
    if (muted_) { setState(Idle); return; }
    backend_.speak(answer);
    setState(Speaking);
}

// ---------------------------------------------------------------- window ---
void FloatingWindow::resizeEvent(QResizeEvent*) {
    const QRect card = rect().adjusted(kMargin, kMargin, -kMargin, -kMargin);
    soundBtn_->move(card.left() + 14, card.top() + 14);
    closeBtn_->move(card.right() - 14 - closeBtn_->width() + 1, card.top() + 14);
}

void FloatingWindow::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QRectF card = QRectF(rect()).adjusted(kMargin, kMargin, -kMargin, -kMargin);
    p.setPen(Qt::NoPen);
    for (int i = 0; i < 16; ++i) {           // layered soft shadow
        p.setBrush(QColor(30, 40, 110, 4));
        p.drawRoundedRect(card.adjusted(-i, -i + 5, i, i + 5), kRadius + i, kRadius + i);
    }
    p.setBrush(Qt::white);                   // always white: no dark mode
    p.setPen(QPen(theme::line, 1));
    p.drawRoundedRect(card, kRadius, kRadius);
}

void FloatingWindow::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && windowHandle()) windowHandle()->startSystemMove();
}

void FloatingWindow::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Escape) { close(); return; }
    QWidget::keyPressEvent(e);
}

void FloatingWindow::closeEvent(QCloseEvent* e) {
    if (background_) {   // the daemon keeps listening; x / Esc only tuck the window away
        pinned_ = false;
        hideTimer_.stop();
        hide();
        e->ignore();
        return;
    }
    mic_.setMode(MicCapture::Off);
    backend_.stopSpeaking();
    QWidget::closeEvent(e);
    QApplication::quit();
}
