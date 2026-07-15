#!/bin/sh
# GPU Watchman installer. Host this file at https://rest.yshubham.com/gpu/install.sh.
set -eu

BASE_URL=${GPU_WATCHMAN_BASE_URL:-https://rest.yshubham.com/gpu}
VERSION=${GPU_WATCHMAN_VERSION:-v0.2.0}
INSTALL_DIR=${GPU_WATCHMAN_INSTALL_DIR:-/usr/local/bin}

os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$arch" in
  x86_64|amd64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) printf '%s\n' "gpu-watchman: unsupported architecture: $arch" >&2; exit 1 ;;
esac
case "$os" in
  linux|darwin) ;;
  *) printf '%s\n' "gpu-watchman: unsupported operating system: $os" >&2; exit 1 ;;
esac

for command in curl tar install; do
  command -v "$command" >/dev/null 2>&1 || { printf '%s\n' "gpu-watchman: missing required command: $command" >&2; exit 1; }
done

archive="gpu-watchman_${os}_${arch}.tar.gz"
url="$BASE_URL/releases/$VERSION/$archive"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/gpu-watchman.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM

printf '%s\n' "Installing GPU Watchman $VERSION for $os/$arch"
curl --fail --location --silent --show-error "$url" --output "$tmp/$archive"

if curl --fail --location --silent --show-error "$url.sha256" --output "$tmp/$archive.sha256"; then
  expected=$(awk '{print $1}' "$tmp/$archive.sha256")
  if command -v sha256sum >/dev/null 2>&1; then actual=$(sha256sum "$tmp/$archive" | awk '{print $1}'); else actual=$(shasum -a 256 "$tmp/$archive" | awk '{print $1}'); fi
  [ "$expected" = "$actual" ] || { printf '%s\n' "gpu-watchman: checksum mismatch" >&2; exit 1; }
else
  printf '%s\n' "gpu-watchman: checksum file unavailable; continuing without verification" >&2
fi

tar -xzf "$tmp/$archive" -C "$tmp"
[ -f "$tmp/gpu-watchman" ] || { printf '%s\n' "gpu-watchman: archive does not contain gpu-watchman" >&2; exit 1; }
mkdir -p "$INSTALL_DIR"
install -m 0755 "$tmp/gpu-watchman" "$INSTALL_DIR/gpu-watchman"
printf '%s\n' "Installed $INSTALL_DIR/gpu-watchman"
