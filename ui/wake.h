// Wake-phrase matching on Whisper output. Whisper often hears "Kiba" as
// Keba / Kibba / Keeba / Cuba, so match loosely, but require a greeting
// ("hey", "hi", ...) or the name as the very first word to avoid false triggers.
#pragma once
#include <QRegularExpression>
#include <QString>
#include <QStringList>
#include <algorithm>
#include <vector>

struct WakeMatch {
    bool matched = false;
    QString rest;   // anything said after the name, e.g. "what time is it"
};

namespace wake {

inline int editDistance(const QString& a, const QString& b) {
    std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
    for (int j = 0; j <= b.size(); ++j) prev[j] = j;
    for (int i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (int j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

inline bool isGreeting(const QString& t) {
    static const QStringList g{"hey", "hay", "hi", "hei", "heya", "hello", "ok", "okay", "yo", "a", "hey,"};
    return g.contains(t);
}

inline bool isName(const QString& t) {
    static const QStringList heard{"kiba", "keba", "kibba", "keeba", "kyba", "kibah", "kiva", "kibbah",
                                   "keebah", "kybah", "kebba", "cuba", "kibo", "kiber"};
    if (t.size() < 3 || t.size() > 7) return false;
    return heard.contains(t) || editDistance(t, "kiba") <= 1;
}

}  // namespace wake

inline WakeMatch matchWake(const QString& text) {
    static const QRegularExpression word("[A-Za-z']+");
    struct Tok { QString t; int end; };
    std::vector<Tok> toks;
    auto it = word.globalMatch(text);
    while (it.hasNext()) {
        const auto m = it.next();
        toks.push_back({m.captured().toLower(), int(m.capturedEnd())});
    }

    int nameAt = -1;
    for (int i = 0; i < int(toks.size()) && i < 3; ++i) {
        const QString& t = toks[i].t;
        if (wake::isName(t) && (i == 0 || wake::isGreeting(toks[i - 1].t))) { nameAt = i; break; }
        // "heykiba" run together
        for (const char* g : {"hey", "hay", "hi"}) {
            const QString gs = g;
            if (t.startsWith(gs) && t.size() > gs.size() && wake::isName(t.mid(gs.size()))) { nameAt = i; break; }
        }
        if (nameAt >= 0) break;
    }
    if (nameAt < 0) return {};

    QString rest = text.mid(toks[nameAt].end);
    rest.remove(QRegularExpression("^[\\s,.:;!?\\-–—]+"));
    rest = rest.trimmed();
    if (rest.size() < 3) rest.clear();
    return {true, rest};
}
