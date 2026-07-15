package collector

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseGPUs(t *testing.T) {
	raw := "0, NVIDIA Test, GPU-a, 555.1, 00000000:01:00.0, P0, 60, 45, 100.5, 200.0, 1500, 7000, 1800, 8000, 10000, 5000, 5000, 50, 40, 4, 4, 16, 16, Default, Enabled, Enabled, 0, 0, 0, 0\n"
	gpus, err := parseGPUs(raw)
	if err != nil || len(gpus) != 1 {
		t.Fatalf("parseGPUs() = %v, %v", gpus, err)
	}
	if gpus[0].MemoryUsedMiB != 5000 || !gpus[0].PersistenceMode {
		t.Fatalf("unexpected GPU: %#v", gpus[0])
	}
}

func TestParseGPUsRejectsMalformedCSV(t *testing.T) {
	if _, err := parseGPUs(`0,"unterminated`); err == nil {
		t.Fatal("expected malformed CSV error")
	}
}

func TestCollectRunsFullDriverProtocol(t *testing.T) {
	command := filepath.Join(t.TempDir(), "fake-nvidia-smi")
	script := `#!/bin/sh
case "$*" in
  *--query-gpu=index*) printf '%s\n' '0, NVIDIA Test, GPU-a, 555.1, 00000000:01:00.0, P0, 60, 45, 100.5, 200.0, 1500, 7000, 1800, 8000, 10000, 5000, 5000, 50, 40, 4, 4, 16, 16, Default, Enabled, Enabled, 0, 0, 0, 0' ;;
  *--query-compute-apps*) printf '%s\n' 'GPU-a, 123, inference-server, 2048' ;;
  *--query-graphics-apps*) printf '%s\n' 'GPU-a, 123, inference-server, 2048' ;;
  *mig.mode.current*) printf '%s\n' 'Enabled' ;;
  *sw_power_cap*) printf '%s\n' 'Active' ;;
  *hw_thermal_slowdown*) printf '%s\n' 'Not Active' ;;
  *sw_thermal_slowdown*) printf '%s\n' 'Not Active' ;;
  *hw_power_brake_slowdown*) printf '%s\n' 'Not Active' ;;
  *hw_slowdown*) printf '%s\n' 'Not Active' ;;
  'topo -m') printf '%s\n' 'GPU0 CPU Affinity' ;;
esac
`
	if err := os.WriteFile(command, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	report, err := (Collector{command: command}).Collect()
	if err != nil {
		t.Fatal(err)
	}
	if len(report.GPUs) != 1 {
		t.Fatalf("GPU count = %d", len(report.GPUs))
	}
	gpu := report.GPUs[0]
	if gpu.MIGMode != "Enabled" || len(gpu.Processes) != 1 || gpu.Processes[0].MemoryMiB != 2048 || !strings.Contains(strings.Join(gpu.ThrottleReasons, ","), "power cap") || report.Topology == "" {
		t.Fatalf("unexpected report: %#v", report)
	}
}

func TestCollectReportsMissingDriver(t *testing.T) {
	_, err := (Collector{command: filepath.Join(t.TempDir(), "missing-nvidia-smi")}).Collect()
	if err == nil || !strings.Contains(err.Error(), "was not found") {
		t.Fatalf("unexpected error: %v", err)
	}
}
