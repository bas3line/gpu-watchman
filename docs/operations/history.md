# Report history and JSON

## Stable report streams

**--format json** emits one indented JSON value. **--format ndjson** emits exactly one compact JSON value and newline per result. Repeated non-quiet output rejects pretty JSON because concatenated objects would not be a valid stream.

~~~sh
# One compact point-in-time record
gpu-watchman snapshot --format ndjson > snapshot.ndjson

# One compact record for every live cycle
gpu-watchman top --watch 5s --format ndjson > reports.ndjson

# Record without also writing reports to stdout
gpu-watchman serve \
  --api-token-file /run/secrets/watchman-api-token \
  --history /var/lib/gpu-watchman/history.ndjson
~~~

The JSON/NDJSON distinction also applies to **doctor**, **capacity**, **history**, and **compare**. Machine output never contains ANSI color or progress messages.

History files are created with mode 0600 on Unix. Every append rejects symlinks, non-regular files, owner changes, and an existing target with any group/other permission before writing. Each NDJSON record is limited to 64 MiB during both append and analysis.

## Selection semantics

**--history PATH** appends the same selected and finalized report used by stdout and the exporter. In particular, **--gpu INDEX|UUID,...** is applied before the report is stored: GPU rows and GPU-scoped findings outside that selection are removed, then status and summary are rebuilt. Endpoint and host-level evidence remain.

~~~sh
gpu-watchman serve --gpu 0,GPU-abcd \
  --api-token-file /run/secrets/watchman-api-token \
  --history /var/lib/gpu-watchman/selected.ndjson
~~~

**--all** only controls whether healthy idle rows are shown in human text; it does not change JSON, API, or history data. If an unfiltered node history is required, do not pass **--gpu**.

## Schema v3 and source evidence

Report schema version 3 contains the host/status/summary envelope, selected GPUs and processes, findings and recommendations, topology, Xid events, inference endpoints, and telemetry-source coverage. Host-level findings retain **gpu_index: -1** for compatibility.

Every source record includes its canonical name, state (**ok**, **partial**, **unavailable**, or **skipped**), collection duration, record count, required flag, and a bounded error when applicable. The current canonical names are **nvidia.inventory**, **nvidia.processes**, **nvidia.optional**, **nvidia.topology**, and **kernel.xid**.

Use **--require-source** to mark coverage that must be complete:

~~~sh
gpu-watchman snapshot --format json \
  --require-source inventory,processes,topology > report.json
~~~

A one-shot report exits 2 when a required source is not **ok**, while still emitting the report and source finding. Continuous **top**/**watch** and **serve**/**exporter** keep publishing and recording the violation so operators and alerts retain evidence instead of losing the monitor.

## Analyze

~~~sh
gpu-watchman history reports.ndjson
gpu-watchman history reports.ndjson --format json
gpu-watchman history reports.ndjson --format ndjson
~~~

The analyzer reports:

- sample window and hosts;
- peak per-GPU VRAM, utilization, and temperature;
- endpoint availability and maximum queue/KV pressure;
- peak request and generation-token throughput, error rate, and mean TTFT;
- overall telemetry-source completeness and counts grouped by non-OK source/state;
- critical/warning sample counts and finding-code frequency.

Every peak or availability value in JSON/NDJSON has an explicit sample count. A
signal that was never observed is `null` with a zero count; it is not reported as
a plausible zero. Human output prints **N/A (n=0)** for the same case. A measured
zero remains numeric and has a non-zero sample count, so automation can distinguish
idle capacity from missing telemetry.

The analyzer fails closed on an empty file, malformed JSON, unsupported zero or
future report versions, and a file that mixes otherwise-supported schema versions.
Every record is also checked before aggregation: summary counts/totals and status
must match the contained GPU, process, endpoint, and finding records; GPU and
runtime measurements must have valid finite ranges; histogram and source-state
evidence must be structurally consistent. Errors include the one-based record line
instead of silently skipping untrusted evidence. A completely empty in-memory
`Report::default()` fixture with status `unknown` remains readable for API
compatibility, but populated records must carry their derived health status.

Use **compare** when the question is not “what peaked?” but “what changed between these two points?” It accepts ordinary JSON snapshots or selects the final non-empty report from each NDJSON file.

~~~sh
gpu-watchman compare known-good.json candidate.json --fail-on-regression
gpu-watchman compare before.ndjson after.ndjson --format ndjson
~~~

See the [comparison guide](comparison.md) for matching and regression rules.

## Rotation

GPU Watchman appends but does not rotate. Use logrotate, journald capture, or a telemetry pipeline. Watchman opens the path anew on every sample, so after an external rotator renames the file, the next write goes to the newly created path. Configure the rotator to preserve the service owner and mode 0600; an unsafe replacement is rejected rather than populated with sensitive process records.

## Support bundle

~~~sh
gpu-watchman bundle --output node-07-support.json \
  --probe http://localhost:8000 \
  --probe-token-file /run/secrets/runtime-metrics-token
~~~

The bundle includes one complete report plus doctor checks and version metadata. It uses create-new semantics and refuses to overwrite a prior incident artifact. Review process commands and cgroups before sharing it.
