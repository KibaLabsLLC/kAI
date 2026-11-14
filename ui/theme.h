// Colors sampled from the Kiba orb mascot (bubble rim, flower, tail stripes).
#pragma once
#include <QColor>
#include <QFont>

namespace theme {
inline const QColor cyan    {0x6A, 0xD8, 0xFD};
inline const QColor blue    {0x3E, 0x64, 0xEB};
inline const QColor violet  {0x7E, 0x63, 0xFA};
inline const QColor pink    {0xF2, 0x9E, 0xF5};
inline const QColor navy    {0x14, 0x1E, 0x4F};   // mascot body / text
inline const QColor lavender{0xDC, 0xDE, 0xFD};   // tail stripe
inline const QColor mist    {0xF4, 0xF5, 0xFE};   // soft surfaces
inline const QColor line    {0xE6, 0xE8, 0xFA};
inline const QColor muted   {0x7B, 0x82, 0xB8};

inline QFont font(qreal pt, QFont::Weight w = QFont::Normal) {
    QFont f;
    f.setFamilies({"Inter", "SF Pro Text", "Segoe UI", "Cantarell", "Noto Sans", "sans-serif"});
    f.setPointSizeF(pt);
    f.setWeight(w);
    return f;
}
}  // namespace theme
