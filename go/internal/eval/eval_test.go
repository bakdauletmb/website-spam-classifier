package eval

import (
	"math"
	"testing"
)

func TestMetrics(t *testing.T) {
	var m Metrics
	m.Add(true, 1)  // tp
	m.Add(true, 1)  // tp
	m.Add(true, 0)  // fp
	m.Add(false, 0) // tn
	m.Add(false, 0) // tn
	m.Add(false, 1) // fn
	if m.TP != 2 || m.FP != 1 || m.TN != 2 || m.FN != 1 {
		t.Fatalf("%+v", m)
	}
	close := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	if !close(m.Accuracy(), 4.0/6) || !close(m.Precision(), 2.0/3) || !close(m.Recall(), 2.0/3) || !close(m.F1(), 2.0/3) || !close(m.FalsePositiveRate(), 1.0/3) {
		t.Errorf("metrics: %s", m)
	}
	var empty Metrics
	if empty.F1() != 0 || empty.Accuracy() != 0 {
		t.Error("empty metrics should be zero, not NaN")
	}
}
