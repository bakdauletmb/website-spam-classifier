// Package model implements a small, dependency-free L2-regularized logistic
// regression over sparse string-keyed features. Weights are kept by name so
// that predictions can be explained in terms of the features that drove them.
package model

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"spamclassifier/internal/features"
)

// Model is a trained classifier. It serializes to a single JSON file.
type Model struct {
	Version     int                `json:"version"`
	FeatureMode string             `json:"feature_mode"`
	Threshold   float64            `json:"threshold"`
	Bias        float64            `json:"bias"`
	Weights     map[string]float64 `json:"weights"`
	TrainedAt   string             `json:"trained_at,omitempty"`
	Train       *TrainOptions      `json:"train_options,omitempty"`
	Notes       map[string]any     `json:"notes,omitempty"`
}

// Example is one labeled feature vector; Label is 1 for spam, 0 otherwise.
type Example struct {
	Name  string
	X     features.Vector
	Label int
}

// TrainOptions controls stochastic gradient descent.
type TrainOptions struct {
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	L2           float64 `json:"l2"`
	MinDF        int     `json:"min_df"`     // drop features seen in fewer documents
	PosWeight    float64 `json:"pos_weight"` // loss weight for spam examples (1 = balanced by count)
	Seed         int64   `json:"seed"`
}

// DefaultTrainOptions are the values used to produce the shipped model.
func DefaultTrainOptions() TrainOptions {
	return TrainOptions{Epochs: 30, LearningRate: 0.04, L2: 2e-4, MinDF: 2, PosWeight: 1.0, Seed: 42}
}

// Contribution is one feature's share of a prediction's log-odds.
type Contribution struct {
	Feature string  `json:"feature"`
	Value   float64 `json:"value"`
	Weight  float64 `json:"weight"`
	Effect  float64 `json:"effect"` // Value*Weight
}

func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}

// Train fits a model with SGD. L2 regularization is applied lazily (only to
// the weights touched by each example) which is exact for SGD and keeps each
// step proportional to the example's sparsity rather than the vocabulary.
func Train(examples []Example, mode features.Mode, opt TrainOptions) *Model {
	// Vocabulary with document-frequency pruning.
	df := map[string]int{}
	for _, ex := range examples {
		for f := range ex.X {
			df[f]++
		}
	}
	index := map[string]int{}
	names := []string{}
	for f, c := range df {
		if c >= opt.MinDF {
			index[f] = len(names)
			names = append(names, f)
		}
	}
	// Deterministic ordering regardless of map iteration.
	sort.Strings(names)
	for i, f := range names {
		index[f] = i
	}

	type sparse struct {
		idx []int
		val []float64
		y   float64
	}
	data := make([]sparse, len(examples))
	for i, ex := range examples {
		s := sparse{y: float64(ex.Label)}
		// Iterate features in sorted order: map iteration order is random,
		// and floating-point summation order would otherwise make training
		// non-reproducible.
		for _, f := range ex.X.Names() {
			if j, ok := index[f]; ok {
				s.idx = append(s.idx, j)
				s.val = append(s.val, ex.X[f])
			}
		}
		data[i] = s
	}

	w := make([]float64, len(names))
	lastStep := make([]int, len(names))
	var bias float64
	rng := rand.New(rand.NewSource(opt.Seed))
	order := rng.Perm(len(data))
	step := 0
	flush := func(decay float64) {
		for j := range w {
			if n := step - lastStep[j]; n > 0 {
				w[j] *= math.Pow(decay, float64(n))
				lastStep[j] = step
			}
		}
	}
	for epoch := 0; epoch < opt.Epochs; epoch++ {
		lr := opt.LearningRate / (1 + 0.15*float64(epoch))
		decay := 1 - lr*opt.L2
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		for _, i := range order {
			s := data[i]
			step++
			// Catch up lazily-deferred regularization on touched weights.
			z := bias
			for k, j := range s.idx {
				if n := step - lastStep[j]; n > 0 {
					w[j] *= math.Pow(decay, float64(n))
					lastStep[j] = step
				}
				z += w[j] * s.val[k]
			}
			p := sigmoid(z)
			g := p - s.y
			if s.y == 1 {
				g *= opt.PosWeight
			}
			bias -= lr * g
			for k, j := range s.idx {
				w[j] -= lr * g * s.val[k]
			}
		}
	}
	// Flush remaining regularization so stored weights are consistent.
	flush(1 - opt.LearningRate/(1+0.15*float64(opt.Epochs-1))*opt.L2)

	m := &Model{
		Version:     1,
		FeatureMode: string(mode),
		Threshold:   0.5,
		Bias:        bias,
		Weights:     make(map[string]float64, len(names)),
		TrainedAt:   time.Now().UTC().Format(time.RFC3339),
		Train:       &opt,
	}
	for j, f := range names {
		if w[j] != 0 {
			m.Weights[f] = w[j]
		}
	}
	return m
}

// Score returns P(spam | x).
func (m *Model) Score(x features.Vector) float64 {
	z := m.Bias
	for f, v := range x {
		z += m.Weights[f] * v
	}
	return sigmoid(z)
}

// IsSpam applies the model threshold.
func (m *Model) IsSpam(x features.Vector) bool { return m.Score(x) >= m.Threshold }

// Explain lists the k features with the largest absolute effect on the
// log-odds, most influential first.
func (m *Model) Explain(x features.Vector, k int) []Contribution {
	out := make([]Contribution, 0, len(x))
	for f, v := range x {
		w, ok := m.Weights[f]
		if !ok || w == 0 {
			continue
		}
		out = append(out, Contribution{Feature: f, Value: v, Weight: w, Effect: v * w})
	}
	sort.Slice(out, func(i, j int) bool {
		if math.Abs(out[i].Effect) != math.Abs(out[j].Effect) {
			return math.Abs(out[i].Effect) > math.Abs(out[j].Effect)
		}
		return out[i].Feature < out[j].Feature
	})
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// Save writes the model as indented JSON.
func (m *Model) Save(path string) error {
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// Load reads a model from JSON bytes.
func Load(b []byte) (*Model, error) {
	var m Model
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Weights == nil {
		m.Weights = map[string]float64{}
	}
	if m.Threshold == 0 {
		m.Threshold = 0.5
	}
	if m.FeatureMode == "" {
		m.FeatureMode = string(features.ModeAll)
	}
	return &m, nil
}

// LoadFile reads a model from disk.
func LoadFile(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(b)
}
