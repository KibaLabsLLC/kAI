#include "backend.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QStandardPaths>
#include <QTextStream>
#include <csignal>
#include <unistd.h>

Backend::Backend(QObject* parent) : QObject(parent) {
    exe_ = qEnvironmentVariable("KIBAVOICE_BIN");
    if (exe_.isEmpty()) {
        const QString beside = QCoreApplication::applicationDirPath() + "/kibavoice";
        exe_ = QFile::exists(beside) ? beside : QStandardPaths::findExecutable("kibavoice");
    }
}

Backend::~Backend() { stopSpeaking(); }

QString Backend::dataDir() {
    QString base = qEnvironmentVariable("XDG_DATA_HOME");
    if (base.isEmpty()) base = QDir::homePath() + "/.local/share";
    return base + "/kibavoice";
}

static QString slurp(const QString& path) {
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll()).trimmed() : QString();
}

QList<ChatEntry> Backend::loadHistory(int max) const {
    QList<ChatEntry> out;
    QFile f(dataDir() + "/history.tsv");
    if (!f.open(QIODevice::ReadOnly)) return out;
    QStringList ids;
    while (!f.atEnd()) {
        const QString id = QString::fromUtf8(f.readLine()).section('\t', 0, 0).trimmed();
        if (!id.isEmpty() && id.size() < 24 && id.toLongLong() > 0) ids << id;
    }
    for (int i = qMax(0, ids.size() - max); i < ids.size(); ++i) {
        const QString base = dataDir() + "/history/" + ids[i];
        const QString q = slurp(base + ".q"), a = slurp(base + ".a");
        if (!q.isEmpty() && !a.isEmpty()) out.push_back({ids[i], q, a});
    }
    return out;
}

QString Backend::lastHistoryId() const {
    const auto h = loadHistory(1);
    return h.isEmpty() ? QString() : h.last().id;
}

static bool anyFile(const QString& glob) {
    return !QDir(Backend::dataDir() + "/models").entryList({glob}, QDir::Files).isEmpty();
}
bool Backend::hasLlm() const { return anyFile("*.gguf"); }
bool Backend::hasStt() const { return anyFile("ggml-*.bin"); }

QProcess* Backend::spawn(const QStringList& args, bool ownGroup) {
    auto* p = new QProcess(this);
    if (ownGroup) p->setChildProcessModifier([] { ::setsid(); });  // so we can stop playback too
    p->start(exe_.isEmpty() ? "kibavoice" : exe_, args);
    return p;
}

void Backend::transcribe(const QString& wavPath) {
    if (job_) return;
    job_ = spawn({"transcribe", wavPath});
    connect(job_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) { job_->deleteLater(); job_ = nullptr; emit failed("kibavoice binary not found"); }
    });
    connect(job_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, wavPath](int code, QProcess::ExitStatus) {
        const QString text = QString::fromUtf8(job_->readAllStandardOutput()).trimmed();
        job_->deleteLater(); job_ = nullptr;
        QFile::remove(wavPath);
        if (code == 0 && !text.isEmpty()) emit transcribed(text);
        else emit failed("Couldn't make that out");
    });
}

void Backend::ask(const QString& question) {
    if (job_) return;
    job_ = spawn({"ask", "--no-speak", "--no-notify", question});
    connect(job_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) { job_->deleteLater(); job_ = nullptr; emit failed("kibavoice binary not found"); }
    });
    connect(job_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, question](int code, QProcess::ExitStatus) {
        const QString ans = QString::fromUtf8(job_->readAllStandardOutput()).trimmed();
        job_->deleteLater(); job_ = nullptr;
        if (code == 0 && !ans.isEmpty()) emit answered(question, ans);
        else emit failed("No answer from the model");
    });
}

void Backend::speak(const QString& text) {
    stopSpeaking();
    speech_ = spawn({"say", text}, true);
    connect(speech_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
        speech_->deleteLater(); speech_ = nullptr;
        emit speechFinished();
    });
}

void Backend::replay(const QString& id) {
    bool ok = false;
    id.toLongLong(&ok);
    if (!ok) return;
    stopSpeaking();
    speech_ = spawn({"replay", id}, true);
    connect(speech_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int, QProcess::ExitStatus) {
        speech_->deleteLater(); speech_ = nullptr;
        emit speechFinished();
    });
}

void Backend::stopSpeaking() {
    if (!speech_) return;
    QProcess* p = speech_;
    speech_ = nullptr;
    p->disconnect(this);
    if (p->processId() > 0) ::kill(-pid_t(p->processId()), SIGTERM);  // whole group: piper + player
    p->terminate();
    p->deleteLater();
    emit speechFinished();
}
