// Package dataset joins the label file with the scraped sites on disk and
// provides deterministic stratified splits for evaluation.
package dataset

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Sample is one labeled site.
type Sample struct {
	Name  string
	Path  string
	Label int // 1 = spam, 0 = not spam
}

// Stats describes how labels and files lined up.
type Stats struct {
	LabelLines    int
	UniqueLabels  int
	Conflicts     int
	Files         int
	Matched       int
	UnlabeledFile int
	MissingFile   int
	Spam          int
	Ham           int
}

// LoadLabels parses "<site>\t<label>" lines. Duplicate lines are tolerated;
// conflicting labels for the same site are dropped and counted.
func LoadLabels(path string) (map[string]int, Stats, error) {
	var st Stats
	f, err := os.Open(path)
	if err != nil {
		return nil, st, err
	}
	defer f.Close()
	labels := map[string]int{}
	conflict := map[string]bool{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			return nil, st, fmt.Errorf("%s: malformed line %q", path, line)
		}
		st.LabelLines++
		name := strings.ToLower(fields[0])
		var y int
		switch fields[1] {
		case "1", "spam", "true":
			y = 1
		case "0", "ham", "false":
			y = 0
		default:
			return nil, st, fmt.Errorf("%s: bad label %q for %s", path, fields[1], name)
		}
		if prev, ok := labels[name]; ok && prev != y {
			conflict[name] = true
		}
		labels[name] = y
	}
	if err := sc.Err(); err != nil {
		return nil, st, err
	}
	for name := range conflict {
		delete(labels, name)
	}
	st.Conflicts = len(conflict)
	st.UniqueLabels = len(labels)
	return labels, st, nil
}

// Load returns the labeled samples whose scraped file exists in sitesDir,
// sorted by name for determinism.
func Load(labelsPath, sitesDir string) ([]Sample, Stats, error) {
	labels, st, err := LoadLabels(labelsPath)
	if err != nil {
		return nil, st, err
	}
	entries, err := os.ReadDir(sitesDir)
	if err != nil {
		return nil, st, err
	}
	present := map[string]bool{}
	var samples []Sample
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		st.Files++
		key := strings.ToLower(name)
		present[key] = true
		y, ok := labels[key]
		if !ok {
			st.UnlabeledFile++
			continue
		}
		st.Matched++
		if y == 1 {
			st.Spam++
		} else {
			st.Ham++
		}
		samples = append(samples, Sample{Name: name, Path: filepath.Join(sitesDir, name), Label: y})
	}
	for name := range labels {
		if !present[name] {
			st.MissingFile++
		}
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i].Name < samples[j].Name })
	return samples, st, nil
}

// StratifiedFolds assigns each sample index to one of k folds so that every
// fold has (almost) the same spam/ham ratio as the whole set.
func StratifiedFolds(samples []Sample, k int, seed int64) [][]int {
	rng := rand.New(rand.NewSource(seed))
	byLabel := map[int][]int{}
	for i, s := range samples {
		byLabel[s.Label] = append(byLabel[s.Label], i)
	}
	folds := make([][]int, k)
	for _, label := range []int{0, 1} {
		idx := byLabel[label]
		rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
		for j, i := range idx {
			folds[j%k] = append(folds[j%k], i)
		}
	}
	for _, f := range folds {
		sort.Ints(f)
	}
	return folds
}
