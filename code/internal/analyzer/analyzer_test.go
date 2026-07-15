package analyzer

import (
	"github.com/gpu-watchman/gpu-watchman/internal/model"
	"testing"
)

func TestAnalyzeReportsSeriousMemoryFaults(t *testing.T) {
	findings := Analyze([]model.GPU{{Index: 0, MemoryTotalMiB: 1000, MemoryUsedMiB: 980, TemperatureC: 91, ECCUncorrected: 1, RetiredPages: 2}})
	want := map[string]bool{"vram-critical": true, "temperature-critical": true, "ecc-uncorrected": true, "retired-pages": true}
	for _, finding := range findings {
		delete(want, finding.Code)
	}
	if len(want) != 0 {
		t.Fatalf("missing findings: %#v", want)
	}
}

func TestAnalyzeReportsThrottleAndMIG(t *testing.T) {
	findings := Analyze([]model.GPU{{Index: 0, MemoryTotalMiB: 1000, MIGMode: "Enabled", ThrottleReasons: []string{"software power cap"}}})
	want := map[string]bool{"clock-throttled": true, "mig-enabled": true}
	for _, finding := range findings {
		delete(want, finding.Code)
	}
	if len(want) != 0 {
		t.Fatalf("missing findings: %#v", want)
	}
}
