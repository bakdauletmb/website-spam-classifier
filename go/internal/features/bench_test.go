package features

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"spamclassifier/internal/page"
)

// BenchmarkExtract runs the extractor over the five largest pages found under
// data/sites (see the README section on training your own model); use
// -cpuprofile to find hot regexes. Skipped when no pages are present.
func BenchmarkExtract(b *testing.B) {
	dir := "../../data/sites"
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) == 0 {
		b.Skip("no pages under", dir)
	}
	type sized struct {
		name string
		size int64
	}
	var files []sized
	for _, e := range entries {
		if info, err := e.Info(); err == nil && !e.IsDir() {
			files = append(files, sized{e.Name(), info.Size()})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].size > files[j].size })
	if len(files) > 5 {
		files = files[:5]
	}
	var docs []*page.Document
	for _, f := range files {
		d, err := page.Load(filepath.Join(dir, f.name))
		if err != nil {
			b.Fatal(err)
		}
		docs = append(docs, d)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, d := range docs {
			Extract(d, ModeSignals)
		}
	}
}
