// Package eval computes binary classification metrics.
package eval

import "fmt"

// Metrics summarizes a confusion matrix.
type Metrics struct {
	TP, FP, TN, FN int
}

// Add records one prediction.
func (m *Metrics) Add(predSpam bool, label int) {
	switch {
	case predSpam && label == 1:
		m.TP++
	case predSpam && label == 0:
		m.FP++
	case !predSpam && label == 0:
		m.TN++
	default:
		m.FN++
	}
}

// Merge accumulates another Metrics into m.
func (m *Metrics) Merge(o Metrics) { m.TP += o.TP; m.FP += o.FP; m.TN += o.TN; m.FN += o.FN }

func (m Metrics) Total() int { return m.TP + m.FP + m.TN + m.FN }

func safeDiv(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func (m Metrics) Accuracy() float64  { return safeDiv(m.TP+m.TN, m.Total()) }
func (m Metrics) Precision() float64 { return safeDiv(m.TP, m.TP+m.FP) }
func (m Metrics) Recall() float64    { return safeDiv(m.TP, m.TP+m.FN) }
func (m Metrics) F1() float64 {
	p, r := m.Precision(), m.Recall()
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// FalsePositiveRate is the share of legitimate sites wrongly flagged as spam,
// the error a search engine cares most about.
func (m Metrics) FalsePositiveRate() float64 { return safeDiv(m.FP, m.FP+m.TN) }

func (m Metrics) String() string {
	return fmt.Sprintf("acc=%.3f prec=%.3f rec=%.3f f1=%.3f fpr=%.3f (tp=%d fp=%d tn=%d fn=%d)",
		m.Accuracy(), m.Precision(), m.Recall(), m.F1(), m.FalsePositiveRate(), m.TP, m.FP, m.TN, m.FN)
}
