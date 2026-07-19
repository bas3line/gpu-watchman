#!/bin/sh
# Watchman release installer. The release artifact keeps the historical
# gpu-watchman filename; the public command installed on PATH is watchman.
set -eu
umask 077

fail() {
  printf '%s\n' "watchman: $*" >&2
  exit 1
}

BASE_URL=${WATCHMAN_BASE_URL:-${GPU_WATCHMAN_BASE_URL:-https://github.com/bas3line/gpu-watchman/releases/download}}
BASE_URL=${BASE_URL%/}
VERSION=${WATCHMAN_VERSION:-${GPU_WATCHMAN_VERSION:-v0.8.2}}
INSTALL_DIR=${WATCHMAN_INSTALL_DIR:-${GPU_WATCHMAN_INSTALL_DIR:-/usr/local/bin}}

case "$BASE_URL" in
  https://*) ;;
  *) fail "WATCHMAN_BASE_URL must use HTTPS" ;;
esac
case "$VERSION" in
  v[0-9]*) ;;
  *) fail "WATCHMAN_VERSION must be a release tag beginning with v and a digit" ;;
esac
case "$VERSION" in
  *[!A-Za-z0-9._-]*) fail "WATCHMAN_VERSION contains unsafe characters" ;;
esac
case "$INSTALL_DIR" in
  /*) ;;
  *) fail "WATCHMAN_INSTALL_DIR must be an absolute path" ;;
esac

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$arch" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) printf '%s\n' "watchman: unsupported architecture: $arch" >&2; exit 1 ;;
esac
case "$os" in
  linux|darwin) ;;
  *) printf '%s\n' "watchman: unsupported operating system: $os" >&2; exit 1 ;;
esac

for command in awk curl install tar; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  fail "missing required checksum command: sha256sum or shasum"
fi

archive="gpu-watchman_${os}_${arch}.tar.gz"
url="$BASE_URL/$VERSION/$archive"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/gpu-watchman.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM

printf '%s\n' "Installing Watchman $VERSION for $os/$arch"
curl --fail --location --proto '=https' --proto-redir '=https' --silent --show-error \
  --output "$tmp/$archive" -- "$url"

if curl --fail --location --proto '=https' --proto-redir '=https' --silent --show-error \
  --output "$tmp/$archive.sha256" -- "$url.sha256"; then
  expected=$(awk -v name="$archive" '
    NR == 1 {
      file = $2
      sub(/^\*/, "", file)
      if (NF != 2 || file != name) exit 1
      print $1
      next
    }
    { exit 1 }
    END { if (NR != 1) exit 1 }
  ' "$tmp/$archive.sha256") || fail "malformed checksum file"
  [ "${#expected}" -eq 64 ] || fail "malformed SHA-256 digest"
  case "$expected" in
    *[!0-9A-Fa-f]*) fail "malformed SHA-256 digest" ;;
  esac
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$tmp/$archive" | awk '{print $1}')
  else
    actual=$(shasum -a 256 "$tmp/$archive" | awk '{print $1}')
  fi
  [ "$expected" = "$actual" ] || fail "checksum mismatch"
else
  printf '%s\n' "watchman: checksum file unavailable; refusing an unverified install" >&2
  printf '%s\n' "set WATCHMAN_ALLOW_UNVERIFIED=1 only for a trusted private mirror" >&2
  allow_unverified=${WATCHMAN_ALLOW_UNVERIFIED:-${GPU_WATCHMAN_ALLOW_UNVERIFIED:-0}}
  [ "$allow_unverified" = 1 ] || exit 1
fi

attestation_mode=${WATCHMAN_VERIFY_ATTESTATION:-${GPU_WATCHMAN_VERIFY_ATTESTATION:-auto}}
case "$attestation_mode" in
  auto|required)
    if command -v gh >/dev/null 2>&1 && gh attestation verify --help >/dev/null 2>&1; then
      printf '%s\n' "Verifying GitHub artifact attestation"
      GH_FORCE_TTY=0 gh attestation verify "$tmp/$archive" \
        --repo bas3line/gpu-watchman >/dev/null || fail "artifact attestation verification failed"
    elif [ "$attestation_mode" = required ]; then
      fail "a GitHub CLI with attestation support is required"
    else
      printf '%s\n' "Compatible GitHub CLI not found; continuing with SHA-256 verification"
    fi
    ;;
  disabled) ;;
  *) fail "WATCHMAN_VERIFY_ATTESTATION must be auto, required, or disabled" ;;
esac

member_listing=$(tar -tvzf "$tmp/$archive" gpu-watchman) || fail "cannot inspect release archive"
printf '%s\n' "$member_listing" | awk '
  NR == 1 && substr($1, 1, 1) == "-" { regular = 1; next }
  { regular = 0 }
  END { if (NR != 1 || !regular) exit 1 }
' >/dev/null || fail "archive must contain exactly one regular gpu-watchman file"
tar -xzf "$tmp/$archive" -C "$tmp" gpu-watchman
[ -f "$tmp/gpu-watchman" ] || fail "archive does not contain a regular gpu-watchman file"
[ ! -L "$tmp/gpu-watchman" ] || fail "archive contains a symbolic-link gpu-watchman"
mkdir -p "$INSTALL_DIR"
install -m 0755 "$tmp/gpu-watchman" "$INSTALL_DIR/watchman"
printf '%s\n' "Installed $INSTALL_DIR/watchman"
printf '%s\n' "Run: watchman version"
