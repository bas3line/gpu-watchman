package render

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

func TestActiveGPUsIncludesBusyAndUnhealthyDevices(t *testing.T) {
	gpus := []model.GPU{{Index: 0}, {Index: 1, MemoryUsedMiB: 1}, {Index: 2}}
	active := ActiveGPUs(gpus, []model.Finding{{GPUIndex: 2, Severity: "warning"}})
	if len(active) != 2 || active[0].Index != 1 || active[1].Index != 2 {
		t.Fatalf("unexpected active GPUs: %#v", active)
	}
}

func TestTextRendersOperationalDetails(t *testing.T) {
	var output bytes.Buffer
	Text(&output, model.Report{CollectedAt: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), GPUs: []model.GPU{{Index: 0, Name: "Test", MemoryUsedMiB: 50, MemoryTotalMiB: 100, MIGMode: "Enabled", ThrottleReasons: []string{"power cap"}, Processes: []model.Process{{PID: 1, Name: "server", MemoryMiB: 50, Owner: "user"}}}}, Findings: []model.Finding{{GPUIndex: 0, Severity: "warning", Code: "vram-high", Message: "high"}}, Topology: "GPU0 CPU", Endpoints: []model.Endpoint{{URL: "http://server", Reachable: true}}})
	for _, fragment := range []string{"GPU 0  Test", "Throttle: power cap", "Process 1: server", "Findings", "Topology", "Endpoint http://server"} {
		if !strings.Contains(output.String(), fragment) {
			t.Fatalf("missing %q in %s", fragment, output.String())
		}
	}
}

func TestTextExplainsEmptyActiveSet(t *testing.T) {
	var output bytes.Buffer
	Text(&output, model.Report{CollectedAt: time.Now()})
	if !strings.Contains(output.String(), "No active GPUs") {
		t.Fatalf("unexpected output: %s", output.String())
	}
}
