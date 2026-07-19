#!/bin/sh
# Fail CI when sync-conflict copies or removed implementation artifacts return.
set -eu

find_workspace_paths() {
  /usr/bin/find . \
    -path './.git' -prune -o \
    -path './target' -prune -o \
    -print
}

find_workspace_files() {
  /usr/bin/find . \
    -path './.git' -prune -o \
    -path './target' -prune -o \
    -type f -print
}

find_workspace_directories() {
  /usr/bin/find . \
    -path './.git' -prune -o \
    -path './target' -prune -o \
    -type d -print
}

# Match sync-client copies and editor remnants by basename. Lower-casing in awk
# keeps this check consistent on case-sensitive Linux and case-insensitive macOS.
suspicious_names=$(find_workspace_paths | /usr/bin/awk '
  {
    path = $0
    name = path
    sub(/^.*\//, "", name)
    name = tolower(name)

    numbered_copy = name ~ /([[:space:]_-][0-9]+|[[:space:]_-]?\([0-9]+\))(\.[^.]+)?$/
    named_copy = name ~ /(^|[[:space:]_.(-])(copy|duplicate)([[:space:]_-]+[0-9]+|[[:space:]_-]*\([0-9]+\))?(\.[^.]+)?$/
    editor_remnant = name ~ /(~|\.bak|\.swp|\.swo|\.swx|\.tmp|\.temp|\.orig|\.rej)$/

    if (numbered_copy || named_copy || editor_remnant) {
      print path
    }
  }
')

# `cargo package` always adds this exact root file alongside its provenance
# marker. Keep rejecting `.orig` remnants in the working repository and every
# other path, while allowing the standard generated crate payload to self-test.
if [ -f './.cargo_vcs_info.json' ] && [ -f './Cargo.toml.orig' ]; then
  suspicious_names=$(printf '%s\n' "$suspicious_names" | /usr/bin/awk '$0 != "./Cargo.toml.orig"')
fi

# Two paths that differ only by case cannot coexist on the default macOS
# filesystem and are almost always an accidental cross-platform collision.
case_collisions=$(find_workspace_paths | /usr/bin/awk '
  {
    folded = tolower($0)
    if ((folded in seen) && seen[folded] != $0) {
      print seen[folded]
      print $0
    } else if (!(folded in seen)) {
      seen[folded] = $0
    }
  }
')

removed_implementation=$(find_workspace_files | /usr/bin/awk '
  {
    path = $0
    name = path
    sub(/^.*\//, "", name)
    name = tolower(name)
    if (name ~ /\.go$/ || name == "go.mod" || name == "go.sum" ||
        name == "go.work" || name == "go.work.sum") {
      print path
    }
  }
')

platform_junk=$(find_workspace_files | /usr/bin/awk '
  {
    path = $0
    name = path
    sub(/^.*\//, "", name)
    name = tolower(name)
    if (name == ".ds_store" || name == "thumbs.db" || name == "desktop.ini") {
      print path
    }
  }
')

build_artifacts=$(
  {
    find_workspace_files | /usr/bin/awk '
      {
        path = $0
        name = path
        sub(/^.*\//, "", name)
        name = tolower(name)
        if (name == "gpu-watchman" ||
            name ~ /\.(exe|dll|dylib|a|o|obj|rlib|rmeta|pdb|profraw|gcda|gcno|class|pyc)$/ ||
            name ~ /\.so(\.[0-9]+)*$/ ||
            name ~ /\.(tar\.gz|tgz|zip|deb|rpm|pkg|dmg)$/) {
          print path
        }
      }
    '
    find_workspace_directories | /usr/bin/awk '
      {
        path = $0
        name = path
        sub(/^.*\//, "", name)
        name = tolower(name)
        if (name == "target" || name == "build" || name == "dist" ||
            name == "debug" || name == "release" || name == ".dsym" ||
            name == "cmakefiles" || name == "__pycache__") {
          print path
        }
      }
    '
  }
)

# Executable source or documentation files are suspicious too. The installer
# and maintained shell utilities are the only executable files expected here.
stray_executables=$(
  find_workspace_files | while IFS= read -r path; do
    [ -x "$path" ] || continue
    [ "$path" = './install.sh' ] && continue
    script_path=${path#./scripts/}
    if [ "$script_path" != "$path" ] && [ "${script_path%.sh}" != "$script_path" ]; then
      continue
    fi
    printf '%s\n' "$path"
  done
)

# Publication dependencies are executable code. Require immutable revisions so
# a moved action tag or base-image tag cannot silently change a release build.
if [ -d './.github' ]; then
  unpinned_actions=$(
    /usr/bin/find ./.github -type f \( -name '*.yml' -o -name '*.yaml' \) -exec /usr/bin/awk '
      $1 == "-" && $2 == "uses:" { ref = $3 }
      $1 == "uses:" { ref = $2 }
      ref != "" {
        if (ref !~ /^\.\//) {
          separator = index(ref, "@")
          revision = substr(ref, separator + 1)
          if (separator == 0 || length(revision) != 40 || revision ~ /[^0-9a-f]/) {
            print FILENAME ":" FNR ": " ref
          }
        }
        ref = ""
      }
    ' {} +
  )
else
  unpinned_actions=''
fi

if [ -f './Dockerfile' ]; then
  unpinned_container_bases=$(
    /usr/bin/awk '
      $1 == "FROM" {
        image = $2
        if (image ~ /^--platform=/) image = $3
        marker = "@sha256:"
        position = index(image, marker)
        digest = substr(image, position + length(marker))
        if (position == 0 || length(digest) != 64 || digest ~ /[^0-9a-f]/) {
          print FILENAME ":" FNR ": " image
        }
      }
    ' ./Dockerfile
  )
else
  unpinned_container_bases=''
fi

failed=0
report_group() {
  label=$1
  entries=$2
  if [ -n "$entries" ]; then
    printf '%s\n' "$label" >&2
    printf '%s\n' "$entries" >&2
    failed=1
  fi
}

report_group 'suspicious duplicate, numbered-copy, or editor-remnant paths:' "$suspicious_names"
report_group 'case-insensitive path collisions:' "$case_collisions"
report_group 'removed Go implementation artifacts:' "$removed_implementation"
report_group 'platform metadata:' "$platform_junk"
report_group 'build artifacts outside target/:' "$build_artifacts"
report_group 'unexpected executable files:' "$stray_executables"
report_group 'GitHub Actions not pinned to full commit SHAs:' "$unpinned_actions"
report_group 'container bases not pinned to SHA-256 digests:' "$unpinned_container_bases"

if [ "$failed" -ne 0 ]; then
  printf '%s\n' 'repository hygiene check failed; remove or reconcile the files above' >&2
  exit 1
fi

printf '%s\n' 'repository hygiene check passed'
