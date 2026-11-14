#!/usr/bin/env bash
# kibavoice installer: build, install to ~/.local, download models, add the app launcher,
# and (optionally) run it as an always-on background daemon.
#
#   ./install.sh                 full install: app launcher + always-on daemon
#   ./install.sh --deps          also install build/runtime packages first (needs sudo)
#   ./install.sh --no-service    app launcher only, no daemon
#   ./install.sh --no-models     skip the model download (the app fetches them on first run)
#   ./install.sh --prefix DIR    install somewhere other than ~/.local
#   ./install.sh --uninstall     remove the daemon and launcher (models in ~/.local/share/kibavoice stay)
set -euo pipefail

PREFIX="$HOME/.local"
DEPS=0; SERVICE=1; MODELS=1; UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --deps) DEPS=1 ;;
    --no-service) SERVICE=0 ;;
    --no-models) MODELS=0 ;;
    --uninstall) UNINSTALL=1 ;;
    --prefix) shift; PREFIX="${1:?--prefix needs a directory}" ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$PREFIX/bin/kibavoice"
say() { printf '\n==> %s\n' "$*"; }

if [ "$UNINSTALL" = 1 ]; then
  say "Removing daemon and launcher"
  if [ -x "$BIN" ]; then "$BIN" uninstall-service || true; fi
  rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/applications/kibavoice.desktop" \
        "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps/kibavoice.png" \
        "${XDG_CONFIG_HOME:-$HOME/.config}/autostart/kibavoice.desktop"
  rm -f "$PREFIX/bin/kibavoice" "$PREFIX/bin/kibavoice-ui"
  rm -rf "$PREFIX/libexec/kibavoice" "$PREFIX/share/kibavoice"
  echo "Done. Models and history are still in ~/.local/share/kibavoice (delete by hand if you want them gone)."
  exit 0
fi

if [ "$DEPS" = 1 ]; then
  say "Installing dependencies"
  if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git curl pkg-config qt6-base-dev qt6-wayland \
         alsa-utils pulseaudio-utils libnotify-bin
  elif command -v pacman >/dev/null; then
    sudo pacman -S --needed --noconfirm base-devel cmake git curl qt6-base qt6-wayland alsa-utils libpulse libnotify
  elif command -v dnf >/dev/null; then
    sudo dnf install -y gcc-c++ cmake git curl qt6-qtbase-devel qt6-qtwayland alsa-utils pulseaudio-utils libnotify
  else
    echo "No known package manager found; install cmake, a C++17 compiler, git, curl, Qt6 (Widgets+Network), alsa-utils by hand." >&2
    exit 1
  fi
fi

for t in cmake git curl; do
  command -v "$t" >/dev/null || { echo "missing: $t (try ./install.sh --deps)" >&2; exit 1; }
done

say "Building (the first build compiles llama.cpp and whisper.cpp; this takes a few minutes)"
cmake -S "$HERE" -B "$HERE/build" -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$HERE/build" -j"$(nproc)"

say "Installing to $PREFIX"
cmake --install "$HERE/build"

if [ ! -x "$PREFIX/bin/kibavoice-ui" ]; then
  echo "kibavoice-ui was not built (Qt6 Widgets/Network not found). Install qt6-base-dev (or run ./install.sh --deps) and re-run." >&2
  exit 1
fi

case ":$PATH:" in *":$PREFIX/bin:"*) ;; *) echo "note: $PREFIX/bin is not on your PATH" ;; esac

if [ "$MODELS" = 1 ]; then
  say "Downloading models (~550 MB: LLM, speech-to-text, voice)"
  "$BIN" fetch
fi

say "Adding the app launcher"
"$BIN" install-desktop

if [ "$SERVICE" = 1 ]; then
  say "Installing the always-on daemon"
  "$BIN" install-service
fi

say "Health check"
"$BIN" doctor || true

cat <<EOF

Done.
  Say "Hey Kiba".  Launch the window from your app menu ("Kiba") or: kibavoice-ui
  Hotkey tip: bind  kibavoice-ui --listen  to a key to talk without the wake phrase.
EOF
