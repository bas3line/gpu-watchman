# Security policy

## Supported versions

Security fixes are released for the latest published GPU Watchman minor version. Upgrade to the newest patch before reporting a suspected vulnerability.

## Reporting a vulnerability

Use the repository's private **Security advisories → Report a vulnerability** flow. Do not open a public issue for an unpatched vulnerability.

Include the affected version, operating system, deployment shape, minimal reproduction, impact, and whether the issue requires local GPU/driver access. Never attach production support bundles, history files, configuration, bearer tokens, API keys, prompts, model outputs, or raw process/cgroup data. Replace them with synthetic evidence.

## Security boundary

GPU collection is read-only. The **canary** command is an explicit active request that can consume inference capacity and billable tokens. The embedded HTTP server does not terminate TLS; remote listeners require explicit opt-in and bearer authentication and should remain behind a bounded TLS proxy or service mesh. See the deployment and CLI references for the complete threat model and hardening controls.

## Verifying release artifacts

The versioned installer, release archives, and checksum files have keyless GitHub artifact attestations; archives also have SHA-256 sidecars. Download and verify `install.sh` as shown in the installation guide instead of piping mutable branch content into a shell. The installer validates the exact checksum record and, when a compatible GitHub CLI is available, verifies the archive attestation before extracting the binary. Set `GPU_WATCHMAN_VERIFY_ATTESTATION=required` to refuse installation when attestation verification is unavailable; `disabled` is the explicit opt-out for trusted private mirrors.

Downloaded assets can also be verified directly:

```sh
sha256sum -c gpu-watchman_linux_amd64.tar.gz.sha256
gh attestation verify gpu-watchman_linux_amd64.tar.gz \
  --repo bas3line/gpu-watchman
```

Published container manifests include BuildKit provenance and an SBOM, plus a GitHub attestation bound to the pushed manifest-list digest:

```sh
gh attestation verify oci://ghcr.io/bas3line/gpu-watchman:0.8.0 \
  --repo bas3line/gpu-watchman
```

The publication workflows gate on the shared tests and a RustSec advisory audit. Third-party actions and official container bases are pinned to immutable commits or OCI index digests; Dependabot proposes their scheduled updates for review.
