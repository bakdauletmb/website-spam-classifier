package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatVerdict(t *testing.T) {
	if got := formatVerdict("example.com", false); got != "URL 'example.com' is not spam." {
		t.Errorf("got %q", got)
	}
	if got := formatVerdict("0-0.info", true); got != "URL '0-0.info' is spam." {
		t.Errorf("got %q", got)
	}
}

func TestClassifyRequiresInput(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(nil, &out, &errb); code != 1 || !strings.Contains(errb.String(), "--input-web-site parameter is required") {
		t.Errorf("code=%d stderr=%q", code, errb.String())
	}
}

func TestClassifyEndToEnd(t *testing.T) {
	// A tiny hand-made model: parked template => spam.
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "m.json")
	if err := os.WriteFile(modelPath, []byte(`{"version":1,"feature_mode":"signals","threshold":0.5,"bias":-1,"weights":{"sig:parked_template":3}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	parked := filepath.Join(dir, "parked.example")
	_ = os.WriteFile(parked, []byte(`<body><iframe src="http://mcc.godaddy.com/park/x"></iframe></body>`), 0o644)
	normal := filepath.Join(dir, "normal.example")
	_ = os.WriteFile(normal, []byte(`<title>Normal</title><body><p>A regular page about gardening.</p></body>`), 0o644)

	var out, errb bytes.Buffer
	if code := run([]string{"--input-web-site", parked, "--model", modelPath}, &out, &errb); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if strings.TrimSpace(out.String()) != "URL 'parked.example' is spam." {
		t.Errorf("stdout = %q", out.String())
	}
	out.Reset()
	if code := run([]string{"--input-web-site", normal, "--model", modelPath, "--explain"}, &out, &errb); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.HasPrefix(out.String(), "URL 'normal.example' is not spam.") || !strings.Contains(out.String(), "spam probability") {
		t.Errorf("stdout = %q", out.String())
	}
	out.Reset()
	if code := run([]string{"--input-web-site", normal, "--model", modelPath, "--json"}, &out, &errb); code != 0 || !strings.Contains(out.String(), `"is_spam": false`) {
		t.Errorf("json output: code=%d %q", code, out.String())
	}
}

func TestEmbeddedModelIsTrained(t *testing.T) {
	m, err := loadModel("")
	if err != nil {
		t.Skip("embedded model not trained yet:", err)
	}
	if len(m.Weights) < 100 {
		t.Errorf("embedded model looks too small: %d weights", len(m.Weights))
	}
}
