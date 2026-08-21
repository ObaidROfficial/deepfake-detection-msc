# Changelog

Notable changes to the OBI DeepFake Image Detector. The model itself is
unchanged since 3.0 — every release since has been about making it usable and
honest about its limits, not about accuracy.

Versions follow the usual three-part convention. The middle number moves when
something is added or behaves differently on purpose; the third moves when
something that was meant to work already is repaired. So 3.3 introduced the
desktop application and the macOS bundle, and 3.3.1 only fixes a fault in it.

## 3.3.1 — The installer stopped breaking the thing it had just installed

### Fixed

- **The detector would not start when launched from the installer**, failing
  with `TclError: Can't find a usable init.tcl`. The install itself was
  correct and the desktop shortcut worked; only the tick-box launch at the end
  was broken, which is the worst possible place for it, since that is the
  first thing a new user does.

  PyInstaller's tkinter runtime hook sets `TCL_LIBRARY` and `TK_LIBRARY` to the
  copy of Tcl unpacked inside the frozen installer's temporary folder, so the
  installer can draw its own window. Child processes inherit the environment,
  so the newly installed detector — a different Python, in its own virtual
  environment — went looking for its Tcl inside `_MEIxxxxxx` and found
  nothing. The installer now strips those variables, along with `PYTHONHOME`,
  `PYTHONPATH` and PyInstaller's own `_MEIPASS2` and `_PYI_*` markers, from
  everything it starts. Where PyInstaller had overwritten a variable that
  already had a value, the saved original is restored rather than discarded.

  `Launch Detector.bat` and the macOS bundle launcher clear the two Tcl
  variables as well. That is belt and braces, not the fix — but a launcher
  that cannot be broken by whatever started it is worth the two lines.

### Changed

- Corrected two stale comments: `installer.py` still described the uninstaller
  as a release asset fetched over the network, and `Launch Detector.bat` still
  claimed the console shows a local URL. Both date from before 3.3.

## 3.3 — A real application, and a native macOS install

### Changed

- **The detector is a desktop application now.** It was a Gradio web interface
  that served a local page and opened a browser tab. That meant importing a web
  framework and an ASGI server to show one image and one number, and ten to
  twenty seconds passed before anything appeared — long enough that the launcher
  had to be given a visible console so the wait did not look like a crash.
  Rebuilt in tkinter, which is in the standard library and already used by the
  installer: startup is two to three seconds, the install no longer carries a
  web stack it never used as a web application, and the two windows look like
  one product. Inference is still imported from `predict.py`, so the window and
  the command line tool still cannot drift apart.
- **macOS installs are a proper `.app` bundle** in Applications. The Python
  environment, the weights and the code all live inside it, so removing the
  program is dragging one icon to the Trash. No uninstaller, and no desktop
  icon, because neither is what a Mac user expects.
- **The Windows uninstaller is carried inside the installer** rather than
  downloaded from the release. That removes a network call that could fail
  after everything else had succeeded, and takes a file off the releases page
  that was only confusing there — downloaded on its own it refuses to run,
  because it is meant to sit in the install folder.
- The two platforms now behave differently on purpose: a bundle you drag to the
  Trash on macOS, a folder with an uninstaller on Windows.

### Fixed

- **The application window was unusable.** `tk.Label` measures `width` and
  `height` in characters and lines, not pixels, so asking for a 320-pixel
  preview asked for a box 320 characters wide. It filled the window and pushed
  the results panel and the entire button bar off the bottom edge.
- **The taskbar showed Python's icon.** Windows groups taskbar buttons by
  Application User Model ID, and a plain `python.exe` process inherits the
  interpreter's. Setting an explicit ID fixes it; the window icon alone does not.

## 3.2 — Graphical installer

The headline change: installing this no longer requires a terminal.

### Added

- **Graphical installer and uninstaller**, compiled to standalone executables
  for **Windows and macOS**. They handle the Python environment, the
  dependencies, the model weights and a desktop shortcut, and reverse all of it
  cleanly.
- **Python bootstrap.** If the machine has no suitable interpreter, the
  installer downloads the official Python 3.11.9 from python.org and verifies
  it against the published checksum before running it. Nothing needs to be
  installed beforehand.
- **Application icon**, drawn from the installer's own palette and generated
  from a committed script so it can be re-rendered at any size.
- **`Launch Detector.bat`** — double-click to start the detector from a source
  checkout without opening PowerShell.
- **The uninstaller is now actually shipped.** It is downloaded into the
  install folder alongside the weights. Previously the documentation claimed it
  was installed with the application and it was not, so an installed copy could
  only be removed by hand.

### Fixed

- **The desktop shortcut went to the wrong folder.** `Path.home()/"Desktop"` is
  wrong wherever OneDrive or a roaming profile has redirected the Desktop; the
  shortcut landed in a phantom directory the user never opens. Both installer
  and uninstaller now read the real location from the registry.
- **The installer reported a shortcut it had not created.** `cscript` ran
  without checking its exit code, and success was logged unconditionally. It
  now verifies the file exists and reports where it went.
- **The uninstaller would have deleted a source checkout.** Run from a clone it
  resolved its target as the repository itself — source, report and git history
  included. It now refuses, naming what it found, and checks parent directories
  too so running it from `installer/dist/` is caught as well.
- **Buttons could be pushed off the bottom of both windows** when the body text
  was long. The navigation bar is now anchored to the bottom before the content
  is laid out.
- **The setup scripts checked that Python existed but not which version.** On a
  machine with only 3.14 they proceeded and then failed at the numpy build.
  They now accept 3.9 to 3.13 and explain themselves otherwise.

### Changed

- One name throughout — "OBI DeepFake Image Detector". The previous "AI Image
  Detector" overclaimed, since it only handles GAN-generated faces.
- **Installs are much smaller.** The installer used to extract the entire
  repository archive, so every installation carried the report, the notebooks,
  all 25 figures and the results CSVs. It now copies only the files needed to
  run the detector.
- PyInstaller's scratch directory moved out of the project folder. Building
  inside a synced folder fails when the sync client takes handles on files
  mid-build, and output from an elevated build cannot afterwards be removed by
  a normal one.

### Known limitations

- The executables are unsigned, so Windows SmartScreen and macOS Gatekeeper
  warn on first run. Signing needs a paid certificate.
- **The macOS installer is Apple Silicon only.** PyInstaller builds for the
  architecture it runs on, and no Intel Mac was available to build a second
  binary or a universal one. Intel users should install from source with
  `setup.sh`; only the compiled installer is architecture-specific, not the
  application.
- PyInstaller warns that `--onefile` with a windowed macOS build "clashes with
  macOS's security" and will be an error in version 7.0. It produces a working
  binary today, but the macOS build will need moving to `--onedir` before
  upgrading PyInstaller.
- The installer has not been tested on a machine with no Python at all. That
  path is verified by reasoning and by simulating the version checks, which is
  weaker than running it — and section 7.3 of the report says so.

### Note for anyone rebuilding this

PyInstaller cannot cross-compile, so each platform's binaries have to be built
on that platform. The Windows executables were built on Windows and the macOS
ones on an Apple Silicon Mac. `installer/build.py` and `installer/make_icon.py`
handle the platform differences themselves — including generating `app.icns`
via `iconutil`, which only works on macOS.

## 3.1 — Interface hardening

- Face detection before classification, after a portrait was classified REAL
  with 100% confidence because the face occupied a fifth of the frame.
- Out-of-range warnings now inspect the cropped region actually fed to the
  model rather than the original upload.
- Pinned OpenCV below 5, where `CascadeClassifier` has moved. macOS installed 5
  and lost face detection while Windows kept working.
- Fixed invisible emphasis in the scope banner, which inherited white from the
  dark theme against a light background.

## 3.0 — Graphical interface

- Gradio interface with drag-and-drop upload, a progress indicator and a
  confidence readout.
- Command line classifier for batches, sharing its inference code with the
  interface so the two cannot drift apart.
- One-command setup scripts for Windows and Unix.
- The interface reports its own operating limits, citing measured accuracy for
  the condition rather than a generic caution.
