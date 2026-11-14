// kibavoice-ui: minimalist floating window for kibavoice.
//   kibavoice-ui              open the window and wait for "Hey Kiba"
//   kibavoice-ui --background start hidden; the window pops up when you say "Hey Kiba"
//   kibavoice-ui --listen     open and start listening right away (skip the wake phrase)
//   kibavoice-ui --quit       stop a running instance (the always-on daemon included)
#include <QApplication>
#include <QIcon>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMenu>
#include <QPalette>
#include <QStyleFactory>
#include <QSystemTrayIcon>

#include "theme.h"
#include "window.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("kibavoice");
    QApplication::setDesktopFileName("kibavoice");
    bool listen = false, background = false, quit = false;
    for (int i = 1; i < argc; ++i) {
        const QString a = argv[i];
        if (a == "--listen") listen = true;
        else if (a == "--background") background = true;
        else if (a == "--quit") quit = true;
        else if (a == "-h" || a == "--help") {
            puts("kibavoice-ui [--background] [--listen] [--quit]\n  Say \"Hey Kiba\". Esc closes (hides, in --background mode).");
            return 0;
        }
    }

    // One window at a time: a second launch just wakes the first.
    const QString name = "kibavoice-ui-" + qEnvironmentVariable("USER", "user");
    {
        QLocalSocket probe;
        probe.connectToServer(name);
        if (probe.waitForConnected(200)) {
            probe.write(quit ? "quit\n" : listen ? "listen\n" : "show\n");
            probe.waitForBytesWritten(500);
            return 0;
        }
    }
    if (quit) return 0;   // nothing running, nothing to stop

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

    QApplication::setQuitOnLastWindowClosed(false);   // hidden background window must not quit the app
    FloatingWindow w(background);

    QLocalServer::removeServer(name);
    QLocalServer server;
    server.setSocketOptions(QLocalServer::UserAccessOption);
    server.listen(name);
    QObject::connect(&server, &QLocalServer::newConnection, &w, [&] {
        while (QLocalSocket* s = server.nextPendingConnection()) {
            QObject::connect(s, &QLocalSocket::readyRead, &w, [&w, s] {
                const QByteArray msg = s->readAll();
                if (msg.contains("quit")) QApplication::quit();
                else if (msg.contains("listen")) w.present(true);       // hotkey: tucks away after the answer
                else w.present(false, true);                            // menu launch: stays until closed
                s->deleteLater();
            });
            QObject::connect(s, &QLocalSocket::disconnected, s, &QObject::deleteLater);
        }
    });

    // Daemon mode gets a tray icon (where the desktop has one) so it can be shown, used or quit.
    QSystemTrayIcon* tray = nullptr;
    if (background && QSystemTrayIcon::isSystemTrayAvailable()) {
        tray = new QSystemTrayIcon(QIcon(":/kibavoice.png"), &app);
        auto* menu = new QMenu;
        menu->addAction("Show Kiba", &w, [&w] { w.present(false, true); });
        menu->addAction("Talk to Kiba", &w, [&w] { w.present(true); });
        menu->addSeparator();
        menu->addAction("Quit Kiba", &app, &QApplication::quit);
        tray->setContextMenu(menu);
        tray->setToolTip("Kiba: say “Hey Kiba”");
        QObject::connect(tray, &QSystemTrayIcon::activated, &w, [&w](QSystemTrayIcon::ActivationReason r) {
            if (r == QSystemTrayIcon::Trigger) w.present(false, true);
        });
        tray->show();
    }

    if (!background || listen) w.present(listen, !background);
    return app.exec();
}
