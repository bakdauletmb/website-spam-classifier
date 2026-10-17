package dataset

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAndFolds(t *testing.T) {
	dir := t.TempDir()
	sites := filepath.Join(dir, "sites")
	if err := os.Mkdir(sites, 0o755); err != nil {
		t.Fatal(err)
	}
	labels := "a.com\t1\nb.com\t0\nc.com\t1\nd.com\t0\nd.com\t0\ne.com\t1\nmissing.com\t1\nconflict.com\t1\nconflict.com\t0\n"
	lp := filepath.Join(dir, "training.txt")
	if err := os.WriteFile(lp, []byte(labels), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"a.com", "b.com", "c.com", "d.com", "e.com", "unlabeled.com", "conflict.com", ".DS_Store"} {
		if err := os.WriteFile(filepath.Join(sites, n), []byte("<p>x</p>"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	samples, st, err := Load(lp, sites)
	if err != nil {
		t.Fatal(err)
	}
	if st.LabelLines != 9 || st.UniqueLabels != 6 || st.Conflicts != 1 {
		t.Errorf("label stats = %+v", st)
	}
	if st.Files != 7 || st.Matched != 5 || st.UnlabeledFile != 2 || st.MissingFile != 1 || st.Spam != 3 || st.Ham != 2 {
		t.Errorf("file stats = %+v", st)
	}
	if len(samples) != 5 || samples[0].Name != "a.com" {
		t.Errorf("samples = %+v", samples)
	}
	folds := StratifiedFolds(samples, 2, 1)
	seen := map[int]bool{}
	for _, f := range folds {
		for _, i := range f {
			if seen[i] {
				t.Fatalf("index %d in two folds", i)
			}
			seen[i] = true
		}
	}
	if len(seen) != len(samples) {
		t.Error("folds do not cover all samples")
	}
}

func TestBadLabel(t *testing.T) {
	lp := filepath.Join(t.TempDir(), "l.txt")
	_ = os.WriteFile(lp, []byte("a.com\tmaybe\n"), 0o644)
	if _, _, err := LoadLabels(lp); err == nil {
		t.Error("expected error for bad label")
	}
}
