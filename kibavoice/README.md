# kibavoice

Local, offline voice assistant with a minimalist floating window.
Type or speak; a small llama.cpp model answers, Piper reads it aloud.

The window is a rounded white card (always light, no dark mode): the Kiba orb mascot
on top (tongue-out idle pose, animated while it listens / thinks / speaks), your
previous chats in the middle, a text box + mic button, and a gradient wave at the
bottom that follows your microphone in the mascot's colors.

| Part | What | Fetched automatically |
|------|------|-----------------------|
| LLM  | llama.cpp (linked), Qwen2.5-0.5B-Instruct Q4_K_M, ~400 MB | first question |
| STT  | whisper.cpp `whisper-cli` + `ggml-tiny.en.bin`, ~75 MB | first voice question |
| TTS  | Piper v1.2.0 binary + `en_US-lessac-medium` voice | first spoken answer |
| UI   | Qt6 Widgets (`kibavoice-ui`) | built with the project |

Models land in `~/.local/share/kibavoice/`. Override with `KIBAVOICE_LLM_URL`,
`KIBAVOICE_STT_URL`, `KIBAVOICE_VOICE` (e.g. `en_US-amy-medium`), `KIBAVOICE_SYSTEM`.

## Build

Needs cmake >= 3.18, a C++17 compiler, git, curl, and Qt6 (Widgets + Network):
`qt6-base-dev` on Debian/Ubuntu, `qt6-base` on Arch. Runtime: `alsa-utils` (arecord),
one of paplay / pw-play / aplay.

    cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
    cmake --build build -j
    cmake --install build
    kibavoice fetch            # optional: pre-download everything
    kibavoice doctor           # shows what is missing
    kibavoice install-desktop  # menu entry + icon (also adds a "Talk to Kiba" action)

No Qt? Configure with `-DKIBAVOICE_BUILD_UI=OFF` for the CLI only.

## Use

    kibavoice ui               # open the floating window (same as kibavoice-ui)
    kibavoice-ui --listen      # open and start listening right away

Bind `kibavoice-ui --listen` to a hotkey in your desktop / window manager to talk from
anywhere. Launching it again while it is open just brings the existing window forward.

- **Mic button or the wave:** tap to start; it stops by itself after ~1 s of silence, or
  tap again (square icon) to send right away. Tap the mascot to stop it speaking.
- **Text box:** type and press Enter (the mic turns into a send arrow).
- **Previous chats:** shown in the window, newest at the bottom. Click an answer to hear it again.
- **Speaker icon** (top left) mutes spoken answers, **x** or **Esc** closes. Drag anywhere to move.

## CLI

    kibavoice ask <text> [--no-speak] [--no-notify]
    kibavoice listen [--no-speak] [--no-notify]
    kibavoice transcribe <file.wav>
    kibavoice say <text>
    kibavoice history | show <id> | replay <id>

## Notes

- The UI is a separate process that drives the CLI (`ask`, `transcribe`, `say`, `replay`),
  so the CLI keeps working on its own and the two share the same history.
- Child processes are started with an argv vector, never a shell, so text from the model or
  the microphone cannot inject commands. `KIBAVOICE_REC_CMD` overrides the recorder command.
- On Wayland, "always on top" is up to the compositor; use a window rule if you want it pinned.
- whisper.cpp is a separate process on purpose: it and llama.cpp each bundle their own ggml.
- Piper here is the archived rhasspy/piper v1.2.0 build (MIT). The maintained
  OHF-Voice/piper1-gpl is GPL-3.0 and is the upgrade path.
