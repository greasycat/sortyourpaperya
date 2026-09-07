#!/usr/bin/env bash
#
# Install sortyourpaperya on macOS or Linux.
#
#   ./install.sh                 install `sortyourpaperya` (and `sypy`) on PATH
#   ./install.sh --service       ...and run the watcher in the background
#   ./install.sh --check         check prerequisites and stop
#   ./install.sh --uninstall     take it back off
#
# Everything it needs beyond a Python interpreter is installed into a
# virtualenv inside the project, so nothing is written outside the project, the
# chosen bin directory, the skills directory an agent reads, and — with
# --service — the supervisor's config. The last two get a symlink each.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$PROJECT_DIR/python"
VENV_DIR="${SORTYOURPAPERYA_VENV_DIR:-$PACKAGE_DIR/.venv}"
BIN_DIR="${SORTYOURPAPERYA_BIN_DIR:-$HOME/.local/bin}"

# 3.11 is the floor the package declares. Named interpreters are tried before
# the bare one because a distribution's `python3` is often older than the
# newest it also ships.
MIN_PYTHON="3.11"
CANDIDATE_PYTHONS="python3.14 python3.13 python3.12 python3.11 python3"

WANT_SERVICE=0
CHECK_ONLY=0
UNINSTALL=0

RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  BOLD=$'\033[1m'; RESET=$'\033[0m'
fi

die()  { printf '%serror%s: %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
step() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RESET"; }

usage() {
  cat >&2 <<USAGE
usage: ./install.sh [--service] [--check] [--uninstall]

  --service     also install the background watcher (launchd or systemd --user)
  --check       report what is missing and stop, changing nothing
  --uninstall   remove the \`sortyourpaperya\`/\`sypy\` and skill links, and the service if installed
  -h, --help    this

  SORTYOURPAPERYA_VENV_DIR    where the virtualenv goes    (default $PACKAGE_DIR/.venv)
  SORTYOURPAPERYA_BIN_DIR     where the commands are linked (default \$HOME/.local/bin)
  SORTYOURPAPERYA_SKILLS_DIR  where the agent skill goes   (default \$HOME/.claude/skills)
USAGE
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --service) WANT_SERVICE=1 ;;
    --check) CHECK_ONLY=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) usage ;;
    *) printf 'unknown option: %s\n\n' "$1" >&2; usage ;;
  esac
  shift
done

case "$(uname -s)" in
  Darwin) PLATFORM="macOS" ;;
  Linux)  PLATFORM="Linux" ;;
  *) die "unsupported platform: $(uname -s). This installs on macOS and Linux." ;;
esac

# ---- prerequisites ---------------------------------------------------------

# The first candidate that is both present and new enough. Reported by full
# version rather than by name, since `python3` says nothing about which it is.
find_python() {
  local candidate
  for candidate in $CANDIDATE_PYTHONS; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# The command that installs poppler here. Guessed from the package manager
# rather than the distribution name, which is the thing that actually varies.
poppler_hint() {
  if [ "$PLATFORM" = macOS ]; then
    printf 'brew install poppler\n'
  elif command -v apt-get >/dev/null 2>&1; then
    printf 'sudo apt-get install poppler-utils\n'
  elif command -v dnf >/dev/null 2>&1; then
    printf 'sudo dnf install poppler-utils\n'
  elif command -v pacman >/dev/null 2>&1; then
    printf 'sudo pacman -S poppler\n'
  elif command -v zypper >/dev/null 2>&1; then
    printf 'sudo zypper install poppler-tools\n'
  elif command -v apk >/dev/null 2>&1; then
    printf 'sudo apk add poppler-utils\n'
  else
    printf 'install the poppler utilities for your distribution\n'
  fi
}

venv_hint() {
  if command -v apt-get >/dev/null 2>&1; then
    printf 'sudo apt-get install python3-venv\n'
  elif command -v dnf >/dev/null 2>&1; then
    printf 'sudo dnf install python3-virtualenv\n'
  else
    printf 'install your distribution\x27s python venv package\n'
  fi
}

MISSING=0

check_prerequisites() {
  step "Checking prerequisites on $PLATFORM"

  if ! PYTHON="$(find_python)"; then
    printf '  %s✗%s no Python %s or newer found (tried: %s)\n' \
      "$RED" "$RESET" "$MIN_PYTHON" "$CANDIDATE_PYTHONS"
    MISSING=1
    PYTHON=""
  else
    ok "python $("$PYTHON" -c 'import platform; print(platform.python_version())') at $PYTHON"
  fi

  # Debian and its derivatives ship `venv` as a separate package, so a working
  # python3 is not on its own enough — and the failure without this check is an
  # opaque one from deep inside ensurepip.
  if [ -n "$PYTHON" ]; then
    if "$PYTHON" -c 'import venv, ensurepip' 2>/dev/null; then
      ok "python can create virtualenvs"
    else
      printf '  %s✗%s python cannot create virtualenvs. Install it with:\n      %s\n' \
        "$RED" "$RESET" "$(venv_hint)"
      MISSING=1
    fi
  fi

  # Not fatal: only scanned PDFs need it, and the failure is reported per
  # document rather than stopping a pass.
  if command -v pdftoppm >/dev/null 2>&1; then
    ok "pdftoppm found at $(command -v pdftoppm)"
  else
    warn "pdftoppm not found. Documents with no text layer will be reported as"
    printf '      failures until poppler is installed:\n      %s\n' "$(poppler_hint)"
  fi

  if [ "$WANT_SERVICE" = 1 ] && [ "$PLATFORM" = Linux ]; then
    if command -v systemctl >/dev/null 2>&1; then
      ok "systemctl found, for the background service"
    else
      printf '  %s✗%s --service needs systemd. Without it, run `sortyourpaperya watch` under\n' "$RED" "$RESET"
      printf '      whatever supervisor you use.\n'
      MISSING=1
    fi
  fi

  [ "$MISSING" = 0 ] || die "prerequisites are missing; nothing was changed"
}

# ---- the shell rc a PATH line belongs in -----------------------------------

shell_rc() {
  case "$(basename "${SHELL:-sh}")" in
    zsh)  printf '%s\n' "${ZDOTDIR:-$HOME}/.zshrc" ;;
    bash) [ "$PLATFORM" = macOS ] && printf '%s\n' "$HOME/.bash_profile" \
            || printf '%s\n' "$HOME/.bashrc" ;;
    fish) printf '%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish" ;;
    *)    printf '%s\n' "$HOME/.profile" ;;
  esac
}

path_line() {
  if [ "$(basename "${SHELL:-sh}")" = fish ]; then
    printf 'fish_add_path %s\n' "$BIN_DIR"
  else
    printf 'export PATH="%s:$PATH"\n' "$BIN_DIR"
  fi
}

on_path() {
  case ":$PATH:" in
    *":$BIN_DIR:"* | *":$BIN_DIR/:"*) return 0 ;;
    *) return 1 ;;
  esac
}

# ---- the work --------------------------------------------------------------

do_uninstall() {
  step "Removing the background service"
  SORTYOURPAPERYA_VENV_DIR="$VENV_DIR" "$PACKAGE_DIR/scripts/sortyourpaperya-service" uninstall 2>/dev/null \
    || note_no_service
  step "Removing the command and the agent skill"
  SORTYOURPAPERYA_VENV_DIR="$VENV_DIR" SORTYOURPAPERYA_BIN_DIR="$BIN_DIR" "$PACKAGE_DIR/scripts/sortyourpaperya-path" unwire
  printf '\nThe virtualenv at %s and your library are left in place.\n' "$VENV_DIR"
}

note_no_service() { printf '  no service to remove\n'; }

if [ "$UNINSTALL" = 1 ]; then
  do_uninstall
  exit 0
fi

check_prerequisites
if [ "$CHECK_ONLY" = 1 ]; then
  printf '\nEverything needed is present. Run ./install.sh to install.\n'
  exit 0
fi

step "Installing sortyourpaperya and the agent skill"
SORTYOURPAPERYA_FROM_INSTALLER=1 SORTYOURPAPERYA_PYTHON="$PYTHON" SORTYOURPAPERYA_VENV_DIR="$VENV_DIR" \
  SORTYOURPAPERYA_BIN_DIR="$BIN_DIR" "$PACKAGE_DIR/scripts/sortyourpaperya-path" wire

if [ "$WANT_SERVICE" = 1 ]; then
  step "Installing the background service"
  SORTYOURPAPERYA_VENV_DIR="$VENV_DIR" "$PACKAGE_DIR/scripts/sortyourpaperya-service" install
fi

step "Done"
if on_path; then
  printf '  Run: %ssypy --help%s  (or %ssortyourpaperya --help%s — same command)\n' \
    "$BOLD" "$RESET" "$BOLD" "$RESET"
else
  printf '  %s is not on your PATH. Add it:\n\n' "$BIN_DIR"
  printf "    echo '%s' >> %s\n\n" "$(path_line)" "$(shell_rc)"
  printf '  Then open a new shell, or run %s%s/sypy --help%s now.\n' \
    "$BOLD" "$BIN_DIR" "$RESET"
fi

if [ "$WANT_SERVICE" = 0 ]; then
  printf '\n  To watch a folder in the background:\n'
  printf '    ./install.sh --service      (after declaring a watch — see python/README.md)\n'
fi
