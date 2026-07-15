# Installation

## Requirements

- NVIDIA GPU and working driver
- `nvidia-smi` in `PATH`
- Linux or macOS host
- Go 1.25.6 only when building from source

Verify the driver before installing:

```sh
nvidia-smi
```

## Build From Source

From the repository root:

```sh
cd code
go build -trimpath -ldflags='-s -w' -o gpu-watchman ./cmd/gpu-watchman
install -m 0755 gpu-watchman /usr/local/bin/gpu-watchman
gpu-watchman version
```

## Hosted Installer

When release artifacts are hosted, install with:

```sh
curl -fsSL https://rest.yshubham.com/gpu/install.sh | sh
```

The installer downloads `gpu-watchman_<os>_<arch>.tar.gz` from the configured release path. See [hosting the installer](#hosting-the-installer) below before publishing it.

## Hosting The Installer

Publish these files for each release version, for example `v0.2.0`:

```text
https://rest.yshubham.com/gpu/releases/v0.2.0/gpu-watchman_linux_amd64.tar.gz
https://rest.yshubham.com/gpu/releases/v0.2.0/gpu-watchman_linux_amd64.tar.gz.sha256
https://rest.yshubham.com/gpu/releases/v0.2.0/gpu-watchman_darwin_arm64.tar.gz
https://rest.yshubham.com/gpu/releases/v0.2.0/gpu-watchman_darwin_arm64.tar.gz.sha256
```

The archive must contain one executable named `gpu-watchman`. Create it with:

```sh
GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o gpu-watchman ./code/cmd/gpu-watchman
tar -czf gpu-watchman_linux_amd64.tar.gz gpu-watchman
shasum -a 256 gpu-watchman_linux_amd64.tar.gz > gpu-watchman_linux_amd64.tar.gz.sha256
```

Set `GPU_WATCHMAN_VERSION` to select a release and `GPU_WATCHMAN_INSTALL_DIR` to change the destination:

```sh
GPU_WATCHMAN_VERSION=v0.2.0 GPU_WATCHMAN_INSTALL_DIR=$HOME/.local/bin \
  curl -fsSL https://rest.yshubham.com/gpu/install.sh | sh
```
