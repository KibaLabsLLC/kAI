// Thin wrapper around the `kibavoice` CLI (ask / transcribe / say / replay) plus
// direct reads of the history files it writes. Child processes get argv vectors,
// never a shell, so typed or spoken text cannot inject commands.
#pragma once
#include <QList>
#include <QObject>
#include <QProcess>
#include <QString>

struct ChatEntry {
    QString id, question, answer;
};

class Backend : public QObject {
    Q_OBJECT
public:
    explicit Backend(QObject* parent = nullptr);
    ~Backend() override;

    static QString dataDir();
    QList<ChatEntry> loadHistory(int max = 40) const;
    QString lastHistoryId() const;
    bool hasLlm() const;
    bool hasStt() const;

    void transcribe(const QString& wavPath);
    void ask(const QString& question);
    void speak(const QString& text);
    void replay(const QString& id);
    void stopSpeaking();
    bool speaking() const { return speech_ != nullptr; }

signals:
    void transcribed(const QString& text);
    void answered(const QString& question, const QString& answer);
    void speechFinished();
    void failed(const QString& message);

private:
    QProcess* spawn(const QStringList& args, bool ownGroup = false);
    QString exe_;
    QProcess* job_ = nullptr;     // transcribe / ask
    QProcess* speech_ = nullptr;  // say / replay
};
