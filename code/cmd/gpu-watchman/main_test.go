package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendHistoryWritesOneJSONRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.ndjson")
	if err := appendHistory(path, map[string]string{"status": "ok"}); err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(contents), `"status":"ok"`) || !strings.HasSuffix(string(contents), "\n") {
		t.Fatalf("unexpected history: %q", contents)
	}
}
