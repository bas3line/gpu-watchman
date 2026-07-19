#!/bin/sh
# Keep release tags and every version-bearing distribution artifact in sync.
set -eu

version=$(/usr/bin/awk '
  $0 == "[package]" { package = 1; next }
  package && /^\[/ { exit }
  package && /^version[[:space:]]*=/ {
    value = $0
    sub(/^[^=]*=[[:space:]]*"/, "", value)
    sub(/"[[:space:]]*$/, "", value)
    print value
    exit
  }
' Cargo.toml)

if [ -z "$version" ]; then
  printf '%s\n' 'could not resolve [package].version from Cargo.toml' >&2
  exit 1
fi

expected_tag="v$version"
if [ "${GITHUB_REF_TYPE:-}" = 'tag' ] && [ "${GITHUB_REF_NAME:-}" != "$expected_tag" ]; then
  printf '%s\n' "release tag ${GITHUB_REF_NAME:-<unset>} does not match $expected_tag" >&2
  exit 1
fi

/usr/bin/grep -F "VERSION=\${WATCHMAN_VERSION:-\${GPU_WATCHMAN_VERSION:-$expected_tag}}" install.sh >/dev/null || {
  printf '%s\n' "install.sh default version does not match $expected_tag" >&2
  exit 1
}
/usr/bin/grep -F "## [$version]" CHANGELOG.md >/dev/null || {
  printf '%s\n' "CHANGELOG.md has no $version release heading" >&2
  exit 1
}
/usr/bin/grep -F "bas3line/watchman:$version@sha256:" packaging/kubernetes/daemonset.yaml >/dev/null || {
  printf '%s\n' "Kubernetes image tag does not match $version" >&2
  exit 1
}

printf '%s\n' "version consistency check passed ($version)"
