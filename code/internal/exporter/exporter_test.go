package exporter

import (
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gpu-watchman/gpu-watchman/internal/model"
)

func TestMetricsExposeGPUAndProcessData(t *testing.T) {
	server := New()
	server.Set(model.Report{GPUs: []model.GPU{{Index: 0, UUID: "gpu-1", Name: "Test", MemoryUsedMiB: 42, MemoryTotalMiB: 100, Processes: []model.Process{{PID: 7, Name: "server", MemoryMiB: 42}}}}})
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, httptest.NewRequest("GET", "/metrics", nil))
	body := recorder.Body.String()
	if recorder.Code != 200 || !strings.Contains(body, "gpu_watchman_vram_used_mib") || !strings.Contains(body, `pid="7"`) {
		t.Fatalf("unexpected metrics: %d %s", recorder.Code, body)
	}
}

func TestServerSupportsConcurrentWritesAndScrapes(t *testing.T) {
	server := New()
	var group sync.WaitGroup
	for range 20 {
		group.Add(2)
		go func() { defer group.Done(); server.Set(model.Report{GPUs: []model.GPU{{Index: 0, Name: "test"}}}) }()
		go func() {
			defer group.Done()
			server.Handler().ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/metrics", nil))
		}()
	}
	group.Wait()
}
