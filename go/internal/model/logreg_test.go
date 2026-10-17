package model

import (
	"math"
	"path/filepath"
	"testing"

	"spamclassifier/internal/features"
)

func synthetic() []Example {
	var ex []Example
	for i := 0; i < 40; i++ {
		// spam: parked signal and a token; ham: copyright and nav markers.
		ex = append(ex, Example{Name: "spam", X: features.Vector{"sig:parked_template": 1, "w:domain": 1, "sig:links<=3": 1}, Label: 1})
		ex = append(ex, Example{Name: "ham", X: features.Vector{"sig:has_copyright": 1, "sig:has_nav_markers": 1, "w:home": 1}, Label: 0})
	}
	// A few noisy, uninformative features.
	ex[0].X["w:noise"] = 1
	ex[1].X["w:noise"] = 1
	return ex
}

func TestTrainSeparates(t *testing.T) {
	m := Train(synthetic(), features.ModeAll, DefaultTrainOptions())
	if s := m.Score(features.Vector{"sig:parked_template": 1, "w:domain": 1}); s < 0.9 {
		t.Errorf("spam score = %.3f, want > 0.9", s)
	}
	if s := m.Score(features.Vector{"sig:has_copyright": 1, "w:home": 1}); s > 0.1 {
		t.Errorf("ham score = %.3f, want < 0.1", s)
	}
	if m.Weights["sig:parked_template"] <= 0 || m.Weights["sig:has_copyright"] >= 0 {
		t.Errorf("weights have wrong sign: %+v", m.Weights)
	}
	if math.Abs(m.Weights["w:noise"]) > math.Abs(m.Weights["sig:parked_template"]) {
		t.Error("noise feature should not dominate")
	}
}

func TestMinDFPrunes(t *testing.T) {
	opt := DefaultTrainOptions()
	opt.MinDF = 3
	m := Train(synthetic(), features.ModeAll, opt)
	if _, ok := m.Weights["w:noise"]; ok {
		t.Error("feature seen in 2 docs should be pruned with min-df 3")
	}
}

func TestDeterministic(t *testing.T) {
	a := Train(synthetic(), features.ModeAll, DefaultTrainOptions())
	b := Train(synthetic(), features.ModeAll, DefaultTrainOptions())
	if a.Bias != b.Bias || len(a.Weights) != len(b.Weights) {
		t.Fatal("training is not deterministic")
	}
	for k, v := range a.Weights {
		if b.Weights[k] != v {
			t.Fatalf("weight %s differs", k)
		}
	}
}

func TestExplainOrdering(t *testing.T) {
	m := &Model{Bias: 0, Threshold: 0.5, Weights: map[string]float64{"a": 2, "b": -3, "c": 0.5}}
	got := m.Explain(features.Vector{"a": 1, "b": 1, "c": 1, "unknown": 1}, 2)
	if len(got) != 2 || got[0].Feature != "b" || got[1].Feature != "a" {
		t.Errorf("Explain = %+v", got)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	m := Train(synthetic(), features.ModeAll, DefaultTrainOptions())
	path := filepath.Join(t.TempDir(), "m.json")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	l, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	x := features.Vector{"sig:parked_template": 1}
	if math.Abs(l.Score(x)-m.Score(x)) > 1e-12 {
		t.Error("scores differ after round trip")
	}
	if l.FeatureMode != "all" || l.Threshold != 0.5 {
		t.Errorf("metadata lost: %+v", l)
	}
}

func TestSigmoid(t *testing.T) {
	if sigmoid(0) != 0.5 || sigmoid(100) < 0.999 || sigmoid(-100) > 0.001 {
		t.Error("sigmoid")
	}
}
