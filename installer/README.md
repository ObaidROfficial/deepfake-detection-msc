# Installer

The graphical installer and uninstaller shipped on the
[Releases](https://github.com/ObaidROfficial/deepfake-detection-msc/releases)
page. Most people never need this folder — it is only relevant if you want to
build the executables yourself or understand what they do.

## Why this exists

The command line route (`setup.ps1` / `setup.sh`) assumes the person already
has Python, a terminal, and some idea of what a virtual environment is. That is
a fair assumption for a developer and a poor one for anybody else. These two
programs are for anybody else.

## What the installer does

1. Looks for a Python between 3.9 and 3.13. If there is none, it offers to
   download and install the official Python 3.11.9 from python.org.
2. Asks where to install and whether to add the second model.
3. Downloads the repository archive and copies out **only the runtime files** —
   `RUNTIME_ITEMS` in `installer.py`. The archive also carries the report, the
   notebooks, every figure and the results CSVs, which together dwarf the
   application and are of no use to someone who just wants to classify an image.
4. Creates a virtual environment and installs the CPU build of PyTorch plus the
   other dependencies.
5. Downloads the trained weights from the latest release.
6. Downloads the compiled uninstaller into the install folder. A failure here
   is logged but does not abort the install — a user who cannot uninstall is
   inconvenienced, whereas a failed install leaves them with nothing.
7. Creates a desktop shortcut and offers to launch.

Roughly 700MB is downloaded in total, nearly all of it PyTorch.

## The Python problem

This is the awkward part, and the reason the design looks the way it does.

The compiled installer carries its own Python runtime — that is how PyInstaller
works — so it launches on a machine with no Python at all. But that runtime is
frozen inside the executable. It cannot create a virtual environment and it
cannot run pip. A real interpreter still has to exist somewhere on the machine.

The obvious fix, bundling Python's official embeddable distribution inside the
installer, does not work either: the Windows embeddable package deliberately
excludes tkinter and pip, so it can neither draw this window nor install
anything.

So the installer fetches the official python.org installer and runs it:

- **Windows** — a per-user silent install (`/quiet InstallAllUsers=0
  PrependPath=1`). No administrator rights, no dialogs.
- **macOS** — the `.pkg` only installs system-wide, so `installer(8)` is run
  through `osascript` with administrator privileges. The user sees the standard
  macOS password prompt.

The downloaded file is checked against the checksum published on the python.org
release page before it is executed, and deleted unrun if it does not match.
That checksum is MD5, because MD5 is all python.org lists for 3.11.9 — the
release pages only began publishing SHA-256 for later versions. So this is a
corruption check, not a tamper check: MD5 is collision-broken and proves
nothing about whether a file was deliberately altered. It does catch the
failure that actually happens, which is a truncated download being executed.

## Why 3.11.9 and not something current

3.11.9 is what the whole project was developed and tested on.

It is also, as of April 2024, the last 3.11 release with a binary installer —
the line is security-only now, so this does hand the user a dated interpreter.
3.13 was the alternative and is still getting maintenance releases. Every
dependency here almost certainly publishes cp313 wheels, but "almost certainly"
is not "tested", and a marker hitting a wheel resolution failure halfway
through a 700MB install is a worse outcome than an old interpreter. Matching
the development environment removes the risk.

The upper bound of 3.13 on what the installer will *accept* is not caution for
its own sake. Development originally started on 3.14 and had to be abandoned
because numpy had no wheel for it and fell back to a source build that failed.
Accepting an interpreter that new would reproduce that failure on the user's
machine, at the worst possible moment, with a compiler error they cannot act
on. If the only Python present is 3.14, the installer says so plainly and
offers to install 3.11.9 alongside it, leaving the existing one alone.

One consequence worth knowing about: a newly installed Python is added to
`PATH` for programs started *afterwards*, so the already-running installer will
not see it there. `find_python()` therefore also checks the fixed locations the
official installers use. If that still fails, the error message tells the user
to close and reopen the installer, which does fix it.

## Building

```bash
pip install pyinstaller
python installer/build.py
```

PyInstaller cannot cross-compile, so a Windows `.exe` has to be built on
Windows and a macOS binary on macOS. Output goes to `installer/dist/`, which is
gitignored — the executables are attached to a release rather than committed.

### On macOS

```bash
git pull
python3 -m pip install pyinstaller pillow
python3 installer/make_logo.py      # also writes app.icns, via iconutil
python3 installer/build.py
```

`make_logo.py` must be run on the Mac: `app.icns` is built with `iconutil`,
which does not exist on Windows, so the Windows build has no macOS icon to use.

The two binaries carry no file extension — `OBI-DeepFake-Image-Detector-Setup`
and `-Uninstall` — which is what `installer.py` expects to download on a
non-Windows machine.

GitHub release assets lose their execute bit, so anyone downloading the Setup
binary has to `chmod +x` it before it will run. The installer restores that
itself on the uninstaller after fetching it, so only the Setup binary needs it
by hand. Release assets can be added to an existing release at any time, so the
macOS binaries do not have to hold up publishing the Windows ones.

Each one is around 20MB. They stay that small because only the installer is
compiled and it imports nothing beyond tkinter and the standard library.

Test the result on a machine that has never had this project installed. That is
the only reliable way to catch a dependency the build environment happened to
provide.

## Code signing

These executables are unsigned, so:

- **Windows** SmartScreen shows "Windows protected your PC". *More info* →
  *Run anyway*.
- **macOS** Gatekeeper blocks a double-click entirely. Right-click → *Open* →
  confirm.

Signing requires a paid certificate from Microsoft or Apple, which is outside
the scope of a student project. The warnings are documented in the main README
rather than left to surprise anyone.

## Where things end up

The two platforms differ deliberately, because their conventions differ.

| | Windows | macOS |
|---|---|---|
| Installed to | `%LOCALAPPDATA%\OBI-DeepFake-Image-Detector` | `~/Applications/OBI DeepFake Image Detector.app` |
| Layout | plain folder | `.app` bundle, everything inside `Contents/Resources` |
| Launched by | desktop shortcut → `Launch Detector.bat` | the bundle itself |
| Removed by | the bundled uninstaller | dragging it to the Trash |

The macOS bundle is self-contained: the virtual environment, the weights and
the code all sit inside it. That is what makes drag-to-Trash a complete
uninstall, and it is why macOS ships no uninstaller at all.

## The uninstaller (Windows only)

Compiled separately, then **embedded inside the installer** with `--add-data`
and written out during install. It used to be published as its own release
asset and downloaded at install time; carrying it removes a network call that
could fail after everything else had succeeded, and takes a file off the
releases page that was only confusing there — downloaded on its own it refuses
to run, because it is meant to sit in the install folder.

It has to be the frozen executable rather than the `.py`: an interpreter cannot
delete the files it is currently running from. `build.py` therefore builds it
first and the installer second.

**It refuses to run against a git working copy.** Run it as a script from a
clone and it resolves its target as the repository itself — source, report and
history included. It checks the target and three levels of parent directory for
a `.git` folder, which is the one marker that separates a checkout from an
install, since GitHub's archive strips it. It also refuses on any folder that
does not contain at least one of `app.py`, `venv` or `models`, on the grounds
that a folder it cannot recognise is not one it should empty.

It removes the application folder, the venv, the weights and the shortcut. It
leaves the shared PyTorch and Hugging Face caches alone unless the user ticks
the box, because unrelated Python projects use them. A `kaggle.json`, if one
exists, is never touched.

A running executable cannot delete itself on Windows, so the final cleanup is
handed to a detached shell that waits three seconds for the process to exit.
`is_self()` makes sure the uninstaller never tries to delete the folder it is
running from before that point.
