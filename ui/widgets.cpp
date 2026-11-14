#include "widgets.h"
#include "theme.h"

#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QScrollBar>
#include <cmath>

// =============================================================== Mascot ====
MascotWidget::MascotWidget(QWidget* parent) : QWidget(parent) {
    idle_.load(":/mascot_idle.png");
    sheet_.load(":/mascot_sheet.png");
    setCursor(Qt::PointingHandCursor);
    clock_.start();
    tick_.setInterval(16);
    connect(&tick_, &QTimer::timeout, this, [this] { update(); });
    tick_.start();
}

void MascotWidget::setActive(bool a) { active_ = a; clock_.restart(); update(); }

void MascotWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    const double t = clock_.elapsed() / 1000.0;
    const double bob = active_ ? 0 : std::sin(t * 1.7) * 3.5;
    const int drawH = 126;
    const double cx = width() / 2.0, floorY = height() - 14;

    // soft ground glow that breathes with the float
    const double gw = 70 - bob * 2.0;
    QRadialGradient g(QPointF(cx, floorY), gw);
    g.setColorAt(0, QColor(62, 100, 235, 46));
    g.setColorAt(1, QColor(62, 100, 235, 0));
    p.setPen(Qt::NoPen);
    p.setBrush(g);
    p.drawEllipse(QPointF(cx, floorY), gw, 9);

    const QPixmap* src = &idle_;
    QRect sr = idle_.rect();
    if (active_ && !sheet_.isNull()) {
        const int fw = sheet_.width() / cols_, fh = sheet_.height() / rows_;
        const int f = int(clock_.elapsed() / frameMs_) % (cols_ * rows_);
        sr = QRect((f % cols_) * fw, (f / cols_) * fh, fw, fh);
        src = &sheet_;
    }
    const double s = double(drawH) / sr.height();
    const QSizeF sz(sr.width() * s, drawH);
    const QPointF tl(cx - sz.width() / 2, floorY - 10 - sz.height() - bob - 6);
    p.drawPixmap(QRectF(tl, sz), *src, QRectF(sr));
}

// ================================================================= Wave ====
WaveWidget::WaveWidget(QWidget* parent) : QWidget(parent) {
    setCursor(Qt::PointingHandCursor);
    clock_.start();
    tick_.setInterval(16);
    connect(&tick_, &QTimer::timeout, this, [this] { update(); });
    tick_.start();
}

void WaveWidget::setLevel(float rms, float lo, float mid, float hi) {
    target_ = rms; w_[0] = lo; w_[1] = mid; w_[2] = hi;
}

void WaveWidget::paintEvent(QPaintEvent*) {
    const qint64 now = clock_.elapsed();
    const double dt = last_ ? std::min(0.05, (now - last_) / 1000.0) : 0.016;
    last_ = now;
    const double t = now / 1000.0;

    // smooth level: fast attack, slower release
    level_ += (target_ - level_) * (target_ > level_ ? 0.45f : 0.12f);
    for (int i = 0; i < 3; ++i) sw_[i] += (w_[i] - sw_[i]) * 0.15f;
    if (mode_ != Listening) target_ = 0;

    double goal = 0.10, speed = 0.8;
    switch (mode_) {
        case Idle: goal = 0.10 + 0.03 * std::sin(t * 0.9); speed = 0.8; break;
        case Listening: goal = 0.10 + 0.9 * level_; speed = 1.6 + 2.0 * level_; break;
        case Thinking: goal = 0.30 + 0.12 * std::sin(t * 3.2); speed = 3.4; break;
        case Speaking: goal = 0.30 + 0.16 * std::fabs(std::sin(t * 4.1) * std::sin(t * 1.7 + 1)); speed = 2.4; break;
    }
    amp_ += (goal - amp_) * 0.14;
    phase_ += dt * speed;

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const double w = width(), h = height();

    // clip to the card's rounded bottom corners
    const double R = 28;
    QPainterPath clip;
    clip.addRoundedRect(QRectF(0, -R, w, h + R), R, R);
    p.setClipPath(clip);

    // gradient drifts slowly sideways; reflects so it never hard-wraps
    const double drift = std::fmod(t * 36.0, w * 2.0);
    QLinearGradient grad(-drift, 0, w * 1.15 - drift, 0);
    grad.setSpread(QGradient::ReflectSpread);
    grad.setColorAt(0.00, theme::cyan);
    grad.setColorAt(0.33, theme::blue);
    grad.setColorAt(0.66, theme::violet);
    grad.setColorAt(1.00, theme::pink);

    struct Layer { double wl, speedMul, alpha, off, band; };
    const Layer layers[4] = {
        {1.15, 0.55, 70, 0.0, sw_[0]}, {0.80, 0.90, 105, 1.7, sw_[1]},
        {0.55, 1.25, 150, 3.1, sw_[2]}, {0.95, 0.75, 255, 4.6, 0.34f}};
    const double baseY = h * 0.60;
    for (int li = 0; li < 4; ++li) {
        const Layer& L = layers[li];
        const double bandBoost = (mode_ == Listening && li < 3) ? (0.55 + 1.5 * L.band) : 1.0;
        const double A = (h * 0.40) * amp_ * bandBoost;
        QPainterPath path;
        path.moveTo(0, h);
        for (double x = 0; x <= w + 4; x += 3) {
            const double u = x / w;
            const double env = std::pow(std::sin(M_PI * std::min(1.0, std::max(0.0, u))), 0.9) * 0.65 + 0.35;
            const double y = baseY - A * env * (std::sin(u * 2 * M_PI / L.wl * 1.6 + phase_ * L.speedMul + L.off)
                                               + 0.35 * std::sin(u * 2 * M_PI * 2.3 - phase_ * 0.7 * L.speedMul + L.off * 2));
            path.lineTo(x, y);
        }
        path.lineTo(w + 4, h);
        path.closeSubpath();
        p.setOpacity(L.alpha / 255.0);
        p.setPen(Qt::NoPen);
        p.setBrush(grad);
        p.drawPath(path);
    }
    p.setOpacity(1.0);

    // keep the very bottom soft so it melts into the card
    QLinearGradient fade(0, h * 0.55, 0, h);
    fade.setColorAt(0, QColor(255, 255, 255, 0));
    fade.setColorAt(1, QColor(255, 255, 255, 60));
    p.setBrush(fade);
    p.drawRect(QRectF(0, 0, w, h));
}

// ============================================================== IconButton ==
IconButton::IconButton(Kind k, int size, QWidget* parent) : QWidget(parent), kind_(k) {
    setFixedSize(size, size);
    setCursor(Qt::PointingHandCursor);
}

void IconButton::mouseReleaseEvent(QMouseEvent* e) {
    const bool was = down_;
    down_ = false;
    update();
    if (was && rect().contains(e->position().toPoint())) emit clicked();
}

void IconButton::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    if (!isEnabled()) p.setOpacity(0.45);
    const QRectF r = QRectF(rect()).adjusted(1, 1, -1, -1);
    const QPointF c = r.center();
    const bool filled = (kind_ == Mic || kind_ == Stop || kind_ == Send);
    const double sc = down_ ? 0.94 : 1.0;
    p.translate(c); p.scale(sc, sc); p.translate(-c);

    if (filled) {
        QLinearGradient g(r.topLeft(), r.bottomRight());
        g.setColorAt(0, theme::cyan); g.setColorAt(0.45, theme::blue); g.setColorAt(1, theme::violet);
        p.setPen(Qt::NoPen);
        if (kind_ == Stop) {      // soft halo while listening
            p.setBrush(QColor(62, 100, 235, busy_ ? 40 : 28));
            p.drawEllipse(r.adjusted(-1, -1, 1, 1));
        }
        p.setBrush(g);
        p.drawEllipse(r.adjusted(2, 2, -2, -2));
    } else if (hover_) {
        p.setPen(Qt::NoPen);
        p.setBrush(theme::mist);
        p.drawEllipse(r);
    }

    const QColor ink = filled ? QColor(Qt::white) : theme::muted;
    QPen pen(ink, filled ? 2.0 : 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    const double u = r.width() / 40.0 * (filled ? 1.0 : 1.25);   // unit scaled to size
    switch (kind_) {
        case Mic: {
            p.setBrush(ink);
            p.drawRoundedRect(QRectF(c.x() - 3.6 * u, c.y() - 9 * u, 7.2 * u, 12.5 * u), 3.6 * u, 3.6 * u);
            p.setBrush(Qt::NoBrush);
            QPainterPath arc; arc.moveTo(c.x() - 7 * u, c.y() - 0.5 * u);
            arc.arcTo(QRectF(c.x() - 7 * u, c.y() - 7 * u, 14 * u, 14 * u), 180, 180);
            p.drawPath(arc);
            p.drawLine(QPointF(c.x(), c.y() + 7 * u), QPointF(c.x(), c.y() + 10.5 * u));
            break;
        }
        case Stop:
            p.setPen(Qt::NoPen); p.setBrush(ink);
            p.drawRoundedRect(QRectF(c.x() - 5.5 * u, c.y() - 5.5 * u, 11 * u, 11 * u), 2.5 * u, 2.5 * u);
            break;
        case Send:
            p.drawLine(QPointF(c.x() - 5 * u, c.y()), QPointF(c.x() + 5 * u, c.y()));
            p.drawLine(QPointF(c.x() + 1 * u, c.y() - 4.5 * u), QPointF(c.x() + 5 * u, c.y()));
            p.drawLine(QPointF(c.x() + 1 * u, c.y() + 4.5 * u), QPointF(c.x() + 5 * u, c.y()));
            break;
        case Close: {
            const double d = 4.2 * u;
            p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d));
            p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d));
            break;
        }
        case SoundOn: case SoundOff: {
            QPainterPath sp;
            sp.moveTo(c.x() - 6 * u, c.y() - 2.5 * u); sp.lineTo(c.x() - 3 * u, c.y() - 2.5 * u);
            sp.lineTo(c.x() + 1 * u, c.y() - 6 * u); sp.lineTo(c.x() + 1 * u, c.y() + 6 * u);
            sp.lineTo(c.x() - 3 * u, c.y() + 2.5 * u); sp.lineTo(c.x() - 6 * u, c.y() + 2.5 * u);
            sp.closeSubpath();
            p.drawPath(sp);
            if (kind_ == SoundOn) {
                p.drawArc(QRectF(c.x() - 1 * u, c.y() - 4 * u, 8 * u, 8 * u), -50 * 16, 100 * 16);
                p.drawArc(QRectF(c.x() - 3 * u, c.y() - 7 * u, 14 * u, 14 * u), -45 * 16, 90 * 16);
            } else {
                p.drawLine(QPointF(c.x() + 4 * u, c.y() - 3 * u), QPointF(c.x() + 9 * u, c.y() + 3 * u));
                p.drawLine(QPointF(c.x() + 4 * u, c.y() + 3 * u), QPointF(c.x() + 9 * u, c.y() - 3 * u));
            }
            break;
        }
    }
}
