# Changelog

Notable changes to the OBI DeepFake Image Detector. The model itself is
unchanged since 3.0 — every release since has been about making it usable and
honest about its limits, not about accuracy.

## 3.2 — Graphical installer

The headline change: installing this no longer requires a terminal.

### Added

- **Graphical installer and uninstaller**, compiled to standalone executables.
  They handle the Python environment, the dependencies, the model weights and a
  desktop shortcut, and reverse all of it cleanly.
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
- macOS binaries must be built on macOS; PyInstaller cannot cross-compile.

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
