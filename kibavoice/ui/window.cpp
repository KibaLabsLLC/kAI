#include "window.h"
#include "theme.h"

#include <QApplication>
#include <QDir>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QPainter>
#include <QScreen>
#include <QSettings>
#include <QVBoxLayout>
#include <QWindow>

namespace {
constexpr int kMargin = 22;   // room for the drop shadow around the card
constexpr int kRadius = 28;
}

FloatingWindow::FloatingWindow(QWidget* parent) : QWidget(parent) {
    setWindowFlags(Qt::Window | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint);
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowTitle("Kiba");
    resize(380 + 2 * kMargin, 610 + 2 * kMargin);
    setMinimumSize(320 + 2 * kMargin, 480 + 2 * kMargin);
    muted_ = QSettings("kibavoice", "ui").value("muted", false).toBool();

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(kMargin, kMargin, kMargin, kMargin);
    root->setSpacing(0);

    mascot_ = new MascotWidget;
    mascot_->setFixedHeight(184);
    root->addWidget(mascot_);

    status_ = new QLabel("Tap the mic to talk");
    status_->setAlignment(Qt::AlignCenter);
    status_->setFont(theme::font(10, QFont::Medium));
    status_->setFixedHeight(24);
    status_->setStyleSheet("color:#7B82B8; background:transparent;");
    root->addWidget(status_);

    chat_ = new ChatView;
    auto* chatRow = new QHBoxLayout;
    chatRow->setContentsMargins(16, 4, 16, 0);
    chatRow->addWidget(chat_);
    root->addLayout(chatRow, 1);

    auto* inputRow = new QHBoxLayout;
    inputRow->setContentsMargins(16, 8, 16, 12);
    inputRow->setSpacing(8);
    input_ = new QLineEdit;
    input_->setPlaceholderText("Ask anything");
    input_->setFixedHeight(40);
    input_->setFont(theme::font(10.5));
    input_->setStyleSheet(
        "QLineEdit{background:#F4F5FE;border:1px solid #E6E8FA;border-radius:20px;padding:0 16px;color:#141E4F;"
        "selection-background-color:#C9D2FF;selection-color:#141E4F;}"
        "QLineEdit:focus{border:1px solid #8FA4FF;}");
    QPalette pp = input_->palette();
    pp.setColor(QPalette::PlaceholderText, theme::muted);
    input_->setPalette(pp);
    micBtn_ = new IconButton(IconButton::Mic, 40);
    inputRow->addWidget(input_, 1);
    inputRow->addWidget(micBtn_);
    root->addLayout(inputRow);

    wave_ = new WaveWidget;
    wave_->setFixedHeight(96);
    root->addWidget(wave_);

    closeBtn_ = new IconButton(IconButton::Close, 30, this);
    soundBtn_ = new IconButton(muted_ ? IconButton::SoundOff : IconButton::SoundOn, 30, this);
    closeBtn_->raise(); soundBtn_->raise();

    // --- wiring ---
    connect(closeBtn_, &IconButton::clicked, this, &QWidget::close);
    connect(soundBtn_, &IconButton::clicked, this, [this] {
        muted_ = !muted_;
        QSettings("kibavoice", "ui").setValue("muted", muted_);
        soundBtn_->setKind(muted_ ? IconButton::SoundOff : IconButton::SoundOn);
        if (muted_ && state_ == Speaking) backend_.stopSpeaking();
    });
    connect(micBtn_, &IconButton::clicked, this, &FloatingWindow::micClicked);
    connect(wave_, &WaveWidget::clicked, this, &FloatingWindow::micClicked);
    connect(mascot_, &MascotWidget::clicked, this, [this] { if (state_ == Speaking) backend_.stopSpeaking(); });
    connect(input_, &QLineEdit::returnPressed, this, &FloatingWindow::submitText);
    connect(input_, &QLineEdit::textChanged, this, &FloatingWindow::updateMicButton);
    connect(chat_, &ChatView::replayRequested, this, [this](const QString& id) {
        if (state_ != Idle && state_ != Speaking) return;
        backend_.replay(id);
        setState(Speaking);
    });

    connect(&mic_, &MicCapture::level, wave_, &WaveWidget::setLevel);
    connect(&mic_, &MicCapture::utteranceReady, this, [this](const QString& wav) {
        setState(Transcribing, backend_.hasStt() ? "Transcribing…" : "First run: downloading speech model…");
        backend_.transcribe(wav);
    });
    connect(&mic_, &MicCapture::nothingHeard, this, [this] { setState(Idle, "Didn't hear anything", true); });
    connect(&mic_, &MicCapture::error, this, [this](const QString& m) { setState(Idle, m, true); });

    connect(&backend_, &Backend::transcribed, this, [this](const QString& text) { askQuestion(text); });
    connect(&backend_, &Backend::answered, this, &FloatingWindow::finishAnswer);
    connect(&backend_, &Backend::failed, this, [this](const QString& m) { setState(Idle, m, true); });
    connect(&backend_, &Backend::speechFinished, this, [this] { if (state_ == Speaking) setState(Idle); });

    statusReset_.setSingleShot(true);
    statusReset_.setInterval(4500);
    connect(&statusReset_, &QTimer::timeout, this, [this] { if (state_ == Idle) setState(Idle); });

    // previous chats
    const auto past = backend_.loadHistory(40);
    for (const auto& e : past) { chat_->addUser(e.question); chat_->addAssistant(e.answer, e.id); }
    chat_->markPreviousChats(!past.isEmpty());

    setState(Idle);
}

void FloatingWindow::present(bool listen) {
    if (!isVisible()) {
        if (QScreen* s = QGuiApplication::primaryScreen()) {
            const QRect a = s->availableGeometry();
            move(a.center().x() - width() / 2, a.top() + int(a.height() * 0.14));
        }
    }
    show();
    raise();
    activateWindow();
    if (listen) startListening(); else input_->setFocus();
}

// ------------------------------------------------------------- state -------
void FloatingWindow::setState(State s, const QString& status, bool isError) {
    state_ = s;
    QString text = status;
    if (text.isEmpty()) {
        switch (s) {
            case Idle: text = "Tap the mic to talk"; break;
            case Listening: text = "Listening…"; break;
            case Transcribing: text = "Transcribing…"; break;
            case Thinking: text = "Thinking…"; break;
            case Speaking: text = "Speaking · tap Kiba to stop"; break;
        }
    }
    status_->setText(text);
    status_->setStyleSheet(QString("color:%1; background:transparent;").arg(isError ? "#D6568F" : "#7B82B8"));
    if (s == Idle && !status.isEmpty()) statusReset_.start(); else statusReset_.stop();

    const bool busy = (s != Idle);
    mascot_->setActive(busy);
    mascot_->setFrameMs(s == Listening ? 170 : s == Speaking ? 190 : 230);
    wave_->setMode(s == Listening ? WaveWidget::Listening
                   : s == Speaking ? WaveWidget::Speaking
                   : (s == Thinking || s == Transcribing) ? WaveWidget::Thinking : WaveWidget::Idle);
    if (s != Listening) wave_->setLevel(0, 0.34f, 0.33f, 0.33f);
    input_->setEnabled(s == Idle);
    if (s == Idle && isVisible() && !input_->hasFocus()) input_->setFocus();
    micBtn_->setEnabled(s == Idle || s == Listening || s == Speaking);
    micBtn_->setBusy(s == Listening);
    updateMicButton();
}

void FloatingWindow::updateMicButton() {
    if (state_ == Listening || state_ == Speaking) micBtn_->setKind(IconButton::Stop);
    else if (state_ == Idle && !input_->text().trimmed().isEmpty()) micBtn_->setKind(IconButton::Send);
    else micBtn_->setKind(IconButton::Mic);
}

// ---------------------------------------------------------- interaction ----
void FloatingWindow::startListening() {
    if (state_ == Speaking) backend_.stopSpeaking();
    if (state_ != Idle && state_ != Speaking) return;
    QDir().mkpath(Backend::dataDir());
    wav_ = Backend::dataDir() + "/tmp-ui-input.wav";
    setState(Listening);
    mic_.start(wav_);
}

void FloatingWindow::micClicked() {
    switch (state_) {
        case Idle:
            if (!input_->text().trimmed().isEmpty()) submitText(); else startListening();
            break;
        case Listening: mic_.finishNow(); break;     // stop & send (or cancel if silent)
        case Speaking: backend_.stopSpeaking(); break;
        default: break;
    }
}

void FloatingWindow::submitText() {
    const QString q = input_->text().trimmed();
    if (q.isEmpty() || state_ != Idle) return;
    input_->clear();
    askQuestion(q);
}

void FloatingWindow::askQuestion(const QString& q) {
    chat_->addUser(q);
    setState(Thinking, backend_.hasLlm() ? "Thinking…" : "First run: downloading language model…");
    backend_.ask(q);
}

void FloatingWindow::finishAnswer(const QString&, const QString& a) {
    chat_->addAssistant(a, backend_.lastHistoryId());
    if (muted_) { setState(Idle); return; }
    backend_.speak(a);
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
    mic_.cancel();
    backend_.stopSpeaking();
    QWidget::closeEvent(e);
    QApplication::quit();
}
