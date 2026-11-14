// kibavoice-ui: minimalist floating window for kibavoice.
//   kibavoice-ui            open the window (or raise the one that is already open)
//   kibavoice-ui --listen   open and start listening right away (bind this to a hotkey)
#include <QApplication>
#include <QIcon>
#include <QLocalServer>
#include <QLocalSocket>
#include <QPalette>
#include <QStyleFactory>

#include "theme.h"
#include "window.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("kibavoice");
    QApplication::setDesktopFileName("kibavoice");
    bool listen = false;
    for (int i = 1; i < argc; ++i) {
        const QString a = argv[i];
        if (a == "--listen") listen = true;
        else if (a == "-h" || a == "--help") {
            puts("kibavoice-ui [--listen]\n  Floating Kiba assistant. Esc closes.");
            return 0;
        }
    }

    // One window at a time: a second launch just wakes the first.
    const QString name = "kibavoice-ui-" + qEnvironmentVariable("USER", "user");
    {
        QLocalSocket probe;
        probe.connectToServer(name);
        if (probe.waitForConnected(200)) {
            probe.write(listen ? "listen\n" : "show\n");
            probe.waitForBytesWritten(500);
            return 0;
        }
    }

    // Always light, whatever the desktop theme says.
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    QPalette pal;
    pal.setColor(QPalette::Window, Qt::white);
    pal.setColor(QPalette::WindowText, theme::navy);
    pal.setColor(QPalette::Base, theme::mist);
    pal.setColor(QPalette::AlternateBase, Qt::white);
    pal.setColor(QPalette::Text, theme::navy);
    pal.setColor(QPalette::Button, theme::mist);
    pal.setColor(QPalette::ButtonText, theme::navy);
    pal.setColor(QPalette::PlaceholderText, theme::muted);
    pal.setColor(QPalette::Highlight, QColor(0xC9, 0xD2, 0xFF));
    pal.setColor(QPalette::HighlightedText, theme::navy);
    QApplication::setPalette(pal);
    QApplication::setFont(theme::font(10.5));
    QApplication::setWindowIcon(QIcon(":/kibavoice.png"));

    FloatingWindow w;

    QLocalServer::removeServer(name);
    QLocalServer server;
    server.setSocketOptions(QLocalServer::UserAccessOption);
    server.listen(name);
    QObject::connect(&server, &QLocalServer::newConnection, &w, [&] {
        while (QLocalSocket* s = server.nextPendingConnection()) {
            QObject::connect(s, &QLocalSocket::readyRead, &w, [&w, s] {
                w.present(s->readAll().contains("listen"));
                s->deleteLater();
            });
            QObject::connect(s, &QLocalSocket::disconnected, s, &QObject::deleteLater);
        }
    });

    w.present(listen);
    return app.exec();
}
