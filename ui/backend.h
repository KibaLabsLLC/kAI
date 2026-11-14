// Thin wrapper around the `kibavoice` CLI (fetch / transcribe / ask / say). Child processes get argv vectors,
// never a shell, so typed or spoken text cannot inject commands.
#pragma once
#include <QList>
#include <QObject>
#include <QProcess>
#include <QString>

class Backend : public QObject {
    Q_OBJECT
public:
    explicit Backend(QObject* parent = nullptr);
    ~Backend() override;

    static QString dataDir();
    bool hasLlm() const;
    bool hasStt() const;

    bool busy() const { return job_ != nullptr; }
    void fetchModels();                                  // kibavoice fetch (first run)
    void transcribe(const QString& wavPath, bool wake);  // wake=true: fast "Hey Kiba" check
    void ask(const QString& question);
    void speak(const QString& text);
    void stopSpeaking();
    bool speaking() const { return speech_ != nullptr; }

signals:
    void transcribed(const QString& text);
    void answered(const QString& question, const QString& answer);
    void speechFinished();
    void fetched(bool ok);
    void failed(const QString& message);

private:
    QProcess* spawn(const QStringList& args, bool ownGroup = false);
    QString exe_;
    QProcess* job_ = nullptr;     // transcribe / ask
    QProcess* speech_ = nullptr;  // say / replay
};
