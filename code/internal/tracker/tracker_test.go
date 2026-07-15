package tracker

import (
	"github.com/gpu-watchman/gpu-watchman/internal/model"
	"testing"
	"time"
)

func TestObserveFindsVRAMGrowth(t *testing.T) {
	tracker := New()
	now := time.Now()
	tracker.Observe([]model.GPU{{Index: 0, UUID: "gpu", Processes: []model.Process{{PID: 1, Name: "server", MemoryMiB: 100}}}}, now)
	got := tracker.Observe([]model.GPU{{Index: 0, UUID: "gpu", Processes: []model.Process{{PID: 1, Name: "server", MemoryMiB: 400}}}}, now.Add(time.Second))
	if len(got) != 1 || got[0].Code != "vram-growth" {
		t.Fatalf("unexpected findings: %#v", got)
	}
}
