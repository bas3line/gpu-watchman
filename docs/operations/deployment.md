# Production deployment runbook

GPU Watchman is read-only, but it observes sensitive process identity and should be deployed like a node-level operations agent.

## Preflight

1. Confirm the host NVIDIA driver and **nvidia-smi** work in the exact service/container context.
2. Run **gpu-watchman doctor** with every inference endpoint the service will probe.
3. Run one detailed JSON snapshot and verify GPU count, UUIDs, process visibility, endpoint runtime detection, and expected optional capabilities.
4. Decide whether kernel-log Xid access is permitted. Set **--no-xid** when it is intentionally unavailable.
5. Decide which evidence is mandatory and test the same **--require-source** policy the service will use.
6. Set API and probe tokens from a secret file or environment injection, never a checked-in unit or manifest.

## Recommended service shape

~~~sh
gpu-watchman serve --interval 5s \
  --listen 127.0.0.1:9400 \
  --probe http://127.0.0.1:8000 \
  --api-token-file /run/secrets/watchman-api-token \
  --probe-token-file /run/secrets/runtime-metrics-token \
  --require-source inventory,processes \
  --history /var/lib/gpu-watchman/history.ndjson \
  --freshness 30s
~~~

**serve** is quiet by default. Add **--emit text** or **--emit ndjson** only when the supervisor should also capture report output.

Bind to loopback unless an authenticated service mesh or TLS reverse proxy owns the external trust boundary. A non-loopback bind requires both **--allow-remote-listen** and a resolved API token; either one without the other fails before the listener starts. The embedded server intentionally does not terminate TLS. Use **/livez** for liveness and **/healthz** for collection freshness; both minimal probe routes are deliberately unauthenticated and contain no node identity.

## systemd

Start from [the hardened unit](../../packaging/systemd/gpu-watchman.service). It uses a dedicated account, read-only system protections, restricted privileges, a private temporary directory, loopback HTTP, and a persistent history directory.

Before enabling it:

- provision the service user and history directory;
- create **/etc/gpu-watchman/api-token** as a root-owned mode-0400 file; the unit injects it with **LoadCredential** and refuses to start without it;
- update inference endpoints and freshness for the node;
- validate whether process/cgroup and kernel-log restrictions match your incident requirements.

## Kubernetes

Start from [the DaemonSet](../../packaging/kubernetes/daemonset.yaml). It expects NVIDIA Container Toolkit, an NVIDIA runtime class, all visible devices, the **utility** driver capability, and host PID visibility for host-process attribution.

Create the required API token Secret before applying the manifest, preferably from an already protected file so the value is not placed in a command argument:

~~~sh
kubectl -n monitoring create secret generic gpu-watchman-api \
  --from-file=api-token=/secure/path/watchman-api-token
~~~

The manifest projects that key at **/run/secrets/gpu-watchman/api-token** with mode 0440 and an **fsGroup** that lets the non-root UID/GID 65532 read it. It disables service-account-token automounting, uses the runtime-default seccomp profile, and runs without added capabilities. The safe base binds **127.0.0.1:9400** inside the pod and intentionally defines no Service, remote-listener opt-in, or HTTP probes. Replacing the image digest alone therefore cannot expose node/process telemetry. If the Secret is absent, empty, unsafe, or unreadable, Watchman also fails closed. Rotate the Secret according to your cluster procedure and restart the DaemonSet so each process reads the new value.

This file is an isolated base, not a complete remote-scraping deployment. It is deliberately fail-closed until its all-zero image digest is replaced with the published **sha256** for the exact 0.8.2 image. To scrape it remotely, create a separate overlay that injects an authenticated mTLS sidecar or service mesh, changes Watchman to a non-loopback listener with **--allow-remote-listen**, exposes only the proxy port through a Service, adds proxy-owned health probes and rate limits, and applies a NetworkPolicy restricted to the authenticated scraper. Plain pod-network HTTP is not an acceptable bearer-token boundary.

The base schedules only nodes labeled **nvidia.com/gpu.present=true** and does not tolerate arbitrary taints. Add a narrowly named toleration if your GPU nodes are tainted.

Review before applying:

- the required image digest replacement and your registry pull policy;
- namespace, service account, Pod Security admission, and runtime class;
- whether **hostPID** is acceptable;
- the overlay's scraper-specific NetworkPolicy and proxy-owned kubelet-probe behavior;
- API token injection and the mandatory authenticated mTLS/service-mesh or TLS-proxy boundary;
- whether node-wide GPU visibility conflicts with your device-plugin isolation model.

## Prometheus

Scrape no faster than the Watchman collection interval. The Kubernetes manifest intentionally omits annotation-based discovery because basic pod annotations cannot provide the required bearer credential. Configure the Prometheus scrape job, PodMonitor, or ServiceMonitor to read the **gpu-watchman-api/api-token** Secret and send **Authorization: Bearer ...**; an unauthenticated scrape receives 401.

Import [the example alerts](../../packaging/prometheus/alerts.yaml), then tune durations and thresholds around model load/warm-up behavior. The **up** metric detects target loss; **gpu_watchman_report_available** and last-success age distinguish a running exporter from fresh successful collection. The example source alert joins **gpu_watchman_source_required** with non-OK **gpu_watchman_source_status** so missing evidence is actionable without stopping the observer.

Per-process metrics contain PID/name/owner labels and can churn. Drop **gpu_watchman_process_vram_mib** at ingestion if the identity value does not justify its cardinality; full records remain available from the authenticated report API and history.

## History and rotation

History is append-only NDJSON opened for each sample with mode 0600 on Unix. Watchman does not rotate it. Use logrotate or a telemetry shipper, size storage for your interval, and protect backups because commands/cgroups may expose model or tenant names.

## Upgrade and rollback

1. Verify the archive checksum and run **gpu-watchman version**.
2. Run **doctor** and a one-shot JSON report with the candidate binary.
3. Compare the current and candidate reports with **compare --fail-on-regression**.
4. Replace the binary/image and watch collection freshness, endpoint availability, and finding counts.
5. Keep the prior binary/image digest until the observation window closes.

Report schema changes are versioned. Automations should reject unsupported future schema versions rather than silently guessing.
