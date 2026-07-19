# Installation

## Requirements

- Linux or macOS
- NVIDIA GPU and working driver for live collection
- **nvidia-smi** in PATH
- GitHub CLI with attestation support for the verified hosted installer
- Rust 1.88 or newer only when building from source

The capacity, history, completions, and help commands work without GPU hardware.

## Build from source

~~~sh
cargo build --locked --release
./target/release/gpu-watchman --version
./target/release/gpu-watchman doctor
install -m 0755 target/release/gpu-watchman /usr/local/bin/gpu-watchman
~~~

The release profile enables thin LTO, one codegen unit, symbol stripping, and abort-on-panic.

## Hosted installer

~~~sh
gh release download v0.8.1 \
  --repo bas3line/gpu-watchman \
  --pattern install.sh
gh attestation verify install.sh --repo bas3line/gpu-watchman
sh install.sh
~~~

Install without elevated privileges:

~~~sh
GPU_WATCHMAN_INSTALL_DIR="$HOME/.local/bin" sh install.sh
~~~

The versioned installer is a release asset with a GitHub keyless attestation; verify it before execution as shown. It selects Linux/macOS and amd64/arm64, downloads a release archive, and requires its SHA-256 sidecar to verify. Override the release with **GPU_WATCHMAN_VERSION**. A missing checksum fails closed; **GPU_WATCHMAN_ALLOW_UNVERIFIED=1** exists only for a trusted private mirror.

## Release artifacts

Pushing a version tag triggers release builds for:

~~~text
gpu-watchman_linux_amd64.tar.gz
gpu-watchman_linux_arm64.tar.gz
gpu-watchman_darwin_amd64.tar.gz
gpu-watchman_darwin_arm64.tar.gz
~~~

Every archive receives a .sha256 file. The release workflow uses a locked dependency graph.

## Container

~~~sh
docker build -t gpu-watchman:local .
docker run --rm --gpus all --pid host \
  --mount type=bind,src=/secure/watchman-api-token,dst=/run/secrets/watchman-api-token,readonly \
  --env GPU_WATCHMAN_API_TOKEN_FILE=/run/secrets/watchman-api-token \
  -p 127.0.0.1:9400:9400 gpu-watchman:local
~~~

The final image is non-root and uses rustls with a compiled-in Web PKI root set, but does not include a GPU driver. NVIDIA Container Toolkit must inject the host driver utility and libraries. Ensure the mounted token file is owned by root or container UID 65532, is not group-writable/executable, grants no access to other users, and remains readable by UID/GID 65532. The default container command runs **serve** on **0.0.0.0:9400** with the explicit remote-listener opt-in, requires API authentication and process telemetry, and intentionally marks kernel-log Xid collection skipped. It fails closed when neither **GPU_WATCHMAN_API_TOKEN** nor **GPU_WATCHMAN_API_TOKEN_FILE** resolves to a valid token. The host port is bound to loopback above; use an authenticated TLS proxy and deliberate network policy before publishing it remotely.

## systemd

The example unit runs as a dedicated **gpu-watchman** account, binds the API to loopback, and writes history under **/var/lib/gpu-watchman**.

~~~sh
useradd --system --home-dir /var/lib/gpu-watchman gpu-watchman
usermod -a -G video gpu-watchman
install -d -m 0750 /etc/gpu-watchman
install -m 0400 /secure/watchman-api-token /etc/gpu-watchman/api-token
install -m 0644 packaging/systemd/gpu-watchman.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gpu-watchman
~~~

GPU device group names vary. Confirm **sudo -u gpu-watchman nvidia-smi** before enabling the unit. The packaged service requires the root-owned API token through systemd's credential directory even though it binds to loopback, and applies kernel/control-group/namespace/capability restrictions compatible with its default **--no-xid** policy.

## Shell completions

~~~sh
gpu-watchman completions bash > /etc/bash_completion.d/gpu-watchman
gpu-watchman completions zsh > ~/.zfunc/_gpu-watchman
gpu-watchman completions fish > ~/.config/fish/completions/gpu-watchman.fish
~~~
