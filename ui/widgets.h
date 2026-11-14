#pragma once
#include <QAbstractScrollArea>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <QPixmap>
#include <QTimer>
#include <QWidget>
#include <vector>

// ---- Mascot: idle pose floats gently; the sprite animation plays while busy ----
class MascotWidget : public QWidget {
    Q_OBJECT
public:
    explicit MascotWidget(QWidget* parent = nullptr);
    void setActive(bool active);          // false = idle pose, true = animation
    void setFrameMs(int ms) { frameMs_ = ms; }
    QSize sizeHint() const override { return {320, 170}; }
signals:
    void clicked();
protected:
    void paintEvent(QPaintEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override { emit clicked(); }
private:
    QPixmap idle_, sheet_;
    int cols_ = 4, rows_ = 4, frameMs_ = 230;
    bool active_ = false;
    QElapsedTimer clock_;
    QTimer tick_;
};

// ---- Gradient wave visualizer (mascot colors) ----
class WaveWidget : public QWidget {
    Q_OBJECT
public:
    enum Mode { Idle, Listening, Thinking, Speaking };
    explicit WaveWidget(QWidget* parent = nullptr);
    void setMode(Mode m) { mode_ = m; }
    void setLevel(float rms, float low, float mid, float high);
    QSize sizeHint() const override { return {320, 96}; }
signals:
    void clicked();
protected:
    void paintEvent(QPaintEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override { emit clicked(); }
private:
    Mode mode_ = Idle;
    float target_ = 0, level_ = 0, w_[3] = {0.34f, 0.33f, 0.33f}, sw_[3] = {0.34f, 0.33f, 0.33f};
    double phase_ = 0, amp_ = 0.1;
    QElapsedTimer clock_;
    QTimer tick_;
    qint64 last_ = 0;
};

// ---- Round icon buttons drawn with QPainter (no icon fonts, no dark-mode surprises) ----
class IconButton : public QWidget {
    Q_OBJECT
public:
    enum Kind { Mic, Stop, Close, SoundOn, SoundOff, Send };
    explicit IconButton(Kind k, int size, QWidget* parent = nullptr);
    void setKind(Kind k) { kind_ = k; update(); }
    Kind kind() const { return kind_; }
    void setBusy(bool b) { busy_ = b; update(); }
signals:
    void clicked();
protected:
    void paintEvent(QPaintEvent*) override;
    void enterEvent(QEnterEvent*) override { hover_ = true; update(); }
    void leaveEvent(QEvent*) override { hover_ = false; down_ = false; update(); }
    void mousePressEvent(QMouseEvent*) override { down_ = true; update(); }
    void mouseReleaseEvent(QMouseEvent* e) override;
private:
    Kind kind_;
    bool hover_ = false, down_ = false, busy_ = false;
};

// ---- Conversation list: previous chats + the live exchange ----
class ChatView : public QAbstractScrollArea {
    Q_OBJECT
public:
    explicit ChatView(QWidget* parent = nullptr);
    void clear();
    void addUser(const QString& text);
    void addAssistant(const QString& text, const QString& id = {});
    void setLastAssistantId(const QString& id);
    void markPreviousChats(bool on) { hasPrevious_ = on; relayout(); }
signals:
    void replayRequested(const QString& id);
protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override { relayout(); }
    void mousePressEvent(QMouseEvent* e) override { e->accept(); }
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
private:
    struct Item { bool user; QString text, id; QRect box; };
    void relayout();
    int hit(const QPoint& vp) const;
    void scrollToEnd();
    std::vector<Item> items_;
    int total_ = 0;
    bool hasPrevious_ = false;
};
