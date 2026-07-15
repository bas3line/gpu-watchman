package probe

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCollectParsesInferenceMetricsWithLabelsAndTimestamp(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("vllm:num_requests_running{model=\"x\"} 4 123\nprocess_cpu 1\n"))
	}))
	defer server.Close()
	endpoints := Collect([]string{server.URL + "/"})
	if len(endpoints) != 1 || !endpoints[0].Reachable || endpoints[0].Metrics[`vllm:num_requests_running{model="x"}`] != 4 {
		t.Fatalf("unexpected endpoints: %#v", endpoints)
	}
}

func TestCollectReportsNonSuccessStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "down", http.StatusServiceUnavailable) }))
	defer server.Close()
	endpoint := Collect([]string{server.URL})[0]
	if endpoint.Reachable || endpoint.Failure != "503 Service Unavailable" {
		t.Fatalf("unexpected endpoint: %#v", endpoint)
	}
}
