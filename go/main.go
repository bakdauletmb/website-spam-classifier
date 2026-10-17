// classify_site decides whether a scraped web site is spam.
//
// Usage:
//
//	classify_site --input-web-site PATH [--explain] [--json] [--model FILE] [--threshold T]
//	classify_site train --labels FILE --sites DIR [--out model.json] [options]
//	classify_site eval  --labels FILE --sites DIR [--folds 5] [--features all|signals|tokens] [--verbose]
//
// With no sub-command the tool classifies one site. A trained model is
// embedded in the binary, so classification needs no extra files.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"spamclassifier/internal/dataset"
	"spamclassifier/internal/eval"
	"spamclassifier/internal/features"
	"spamclassifier/internal/model"
	"spamclassifier/internal/page"
)

//go:embed model.json
var embeddedModel []byte

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) > 0 {
		switch args[0] {
		case "train":
			return cmdTrain(args[1:], stdout, stderr)
		case "eval":
			return cmdEval(args[1:], stdout, stderr)
		case "classify":
			args = args[1:]
		}
	}
	return cmdClassify(args, stdout, stderr)
}

// ---------------------------------------------------------------- classify

type result struct {
	URL      string               `json:"url"`
	IsSpam   bool                 `json:"is_spam"`
	Score    float64              `json:"score"`
	Thresh   float64              `json:"threshold"`
	Features int                  `json:"features"`
	Top      []model.Contribution `json:"top_features,omitempty"`
}

func cmdClassify(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("classify_site", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input-web-site", "", "Path to a scraped web site (file or directory) to classify.")
	modelPath := fs.String("model", "", "Model JSON to use instead of the embedded one.")
	threshold := fs.Float64("threshold", 0, "Override the spam probability threshold (0,1).")
	explain := fs.Bool("explain", false, "Print the score and the features that drove the decision.")
	asJSON := fs.Bool("json", false, "Print the result as JSON.")
	topK := fs.Int("top", 12, "Number of features to show with --explain.")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "The --input-web-site parameter is required.")
		return 1
	}
	m, err := loadModel(*modelPath)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	if *threshold > 0 {
		m.Threshold = *threshold
	}
	doc, err := page.Load(*input)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	x := features.Extract(doc, features.Mode(m.FeatureMode))
	r := result{URL: doc.Domain, Score: m.Score(x), Thresh: m.Threshold, Features: len(x)}
	r.IsSpam = r.Score >= m.Threshold
	if *explain || *asJSON {
		r.Top = m.Explain(x, *topK)
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r)
		return 0
	}
	fmt.Fprintln(stdout, formatVerdict(r.URL, r.IsSpam))
	if *explain {
		fmt.Fprintf(stdout, "  spam probability: %.3f (threshold %.2f, %d active features)\n", r.Score, r.Thresh, r.Features)
		fmt.Fprintln(stdout, "  strongest evidence:")
		for _, c := range r.Top {
			fmt.Fprintf(stdout, "    %+6.2f  %s", c.Effect, c.Feature)
			if c.Value != 1 {
				fmt.Fprintf(stdout, " (value %.2f)", c.Value)
			}
			fmt.Fprintln(stdout)
		}
	}
	return 0
}

// formatVerdict produces the one-line output required by the brief.
func formatVerdict(url string, isSpam bool) string {
	label := "is not"
	if isSpam {
		label = "is"
	}
	return fmt.Sprintf("URL '%s' %s spam.", url, label)
}

func loadModel(path string) (*model.Model, error) {
	if path != "" {
		return model.LoadFile(path)
	}
	m, err := model.Load(embeddedModel)
	if err != nil {
		return nil, fmt.Errorf("embedded model: %w", err)
	}
	if len(m.Weights) == 0 {
		return nil, fmt.Errorf("embedded model is empty; run `make train` and rebuild, or pass --model")
	}
	return m, nil
}

// ------------------------------------------------------------------- train

type trainFlags struct {
	labels, sites, mode string
	opt                 model.TrainOptions
}

func addTrainFlags(fs *flag.FlagSet) *trainFlags {
	tf := &trainFlags{opt: model.DefaultTrainOptions()}
	fs.StringVar(&tf.labels, "labels", "data/labels.txt", "Label file: one '<site> <0|1>' per line.")
	fs.StringVar(&tf.sites, "sites", "data/sites", "Directory of scraped sites.")
	fs.StringVar(&tf.mode, "features", string(features.ModeAll), "Feature families to use: all|signals|tokens.")
	fs.IntVar(&tf.opt.Epochs, "epochs", tf.opt.Epochs, "SGD epochs.")
	fs.Float64Var(&tf.opt.LearningRate, "lr", tf.opt.LearningRate, "Initial learning rate.")
	fs.Float64Var(&tf.opt.L2, "l2", tf.opt.L2, "L2 regularization strength.")
	fs.IntVar(&tf.opt.MinDF, "min-df", tf.opt.MinDF, "Drop features seen in fewer than this many sites.")
	fs.Float64Var(&tf.opt.PosWeight, "pos-weight", tf.opt.PosWeight, "Loss weight for spam examples.")
	fs.Int64Var(&tf.opt.Seed, "seed", tf.opt.Seed, "Random seed.")
	return tf
}

func loadExamples(tf *trainFlags, stderr io.Writer) ([]model.Example, features.Mode, error) {
	mode, err := features.ParseMode(tf.mode)
	if err != nil {
		return nil, "", err
	}
	samples, st, err := dataset.Load(tf.labels, tf.sites)
	if err != nil {
		return nil, "", err
	}
	fmt.Fprintf(stderr, "labels: %d lines, %d unique, %d conflicting; files: %d, matched %d (spam %d / ham %d), %d unlabeled files, %d labels without a file\n",
		st.LabelLines, st.UniqueLabels, st.Conflicts, st.Files, st.Matched, st.Spam, st.Ham, st.UnlabeledFile, st.MissingFile)
	start := time.Now()
	examples := make([]model.Example, 0, len(samples))
	for _, s := range samples {
		doc, err := page.Load(s.Path)
		if err != nil {
			fmt.Fprintf(stderr, "skip %s: %v\n", s.Name, err)
			continue
		}
		examples = append(examples, model.Example{Name: s.Name, X: features.Extract(doc, mode), Label: s.Label})
	}
	fmt.Fprintf(stderr, "extracted features for %d sites in %s\n", len(examples), time.Since(start).Round(time.Millisecond))
	return examples, mode, nil
}

func cmdTrain(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tf := addTrainFlags(fs)
	out := fs.String("out", "model.json", "Where to write the trained model.")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	examples, mode, err := loadExamples(tf, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	start := time.Now()
	m := model.Train(examples, mode, tf.opt)
	var train eval.Metrics
	for _, ex := range examples {
		train.Add(m.IsSpam(ex.X), ex.Label)
	}
	m.Notes = map[string]any{
		"training_sites": len(examples),
		"training_fit":   train.String(),
	}
	if err := m.Save(*out); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stdout, "trained %d weights in %s; training-set fit %s\nwrote %s\n", len(m.Weights), time.Since(start).Round(time.Millisecond), train, *out)
	return 0
}

// -------------------------------------------------------------------- eval

func cmdEval(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tf := addTrainFlags(fs)
	folds := fs.Int("folds", 5, "Number of cross-validation folds.")
	threshold := fs.Float64("threshold", 0.5, "Spam probability threshold.")
	verbose := fs.Bool("verbose", false, "List misclassified sites.")
	sweep := fs.Bool("sweep", false, "Also print metrics across thresholds.")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	examples, mode, err := loadExamples(tf, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	samples := make([]dataset.Sample, len(examples))
	for i, ex := range examples {
		samples[i] = dataset.Sample{Name: ex.Name, Label: ex.Label}
	}
	assign := dataset.StratifiedFolds(samples, *folds, tf.opt.Seed)

	type pred struct {
		name  string
		score float64
		label int
	}
	var preds []pred
	var total eval.Metrics
	fmt.Fprintf(stdout, "%d-fold cross-validation, features=%s, threshold=%.2f\n", *folds, mode, *threshold)
	for k, test := range assign {
		inTest := map[int]bool{}
		for _, i := range test {
			inTest[i] = true
		}
		var train []model.Example
		for i, ex := range examples {
			if !inTest[i] {
				train = append(train, ex)
			}
		}
		m := model.Train(train, mode, tf.opt)
		m.Threshold = *threshold
		var fm eval.Metrics
		for _, i := range test {
			s := m.Score(examples[i].X)
			preds = append(preds, pred{examples[i].Name, s, examples[i].Label})
			fm.Add(s >= *threshold, examples[i].Label)
		}
		fmt.Fprintf(stdout, "  fold %d: %s\n", k+1, fm)
		total.Merge(fm)
	}
	fmt.Fprintf(stdout, "overall: %s\n", total)

	if *sweep {
		fmt.Fprintln(stdout, "threshold sweep (pooled out-of-fold predictions):")
		for _, t := range []float64{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} {
			var mt eval.Metrics
			for _, p := range preds {
				mt.Add(p.score >= t, p.label)
			}
			fmt.Fprintf(stdout, "  t=%.1f %s\n", t, mt)
		}
	}
	if *verbose {
		sort.Slice(preds, func(i, j int) bool { return preds[i].score > preds[j].score })
		fmt.Fprintln(stdout, "false positives (legitimate sites flagged as spam), most confident first:")
		for _, p := range preds {
			if p.label == 0 && p.score >= *threshold {
				fmt.Fprintf(stdout, "  %.3f %s\n", p.score, p.name)
			}
		}
		fmt.Fprintln(stdout, "false negatives (spam sites missed), least confident first:")
		for i := len(preds) - 1; i >= 0; i-- {
			p := preds[i]
			if p.label == 1 && p.score < *threshold {
				fmt.Fprintf(stdout, "  %.3f %s\n", p.score, p.name)
			}
		}
	}
	return 0
}

// usage text for -h on the root command.
func init() {
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, strings.TrimSpace(`
classify_site --input-web-site PATH [--explain] [--json] [--model FILE] [--threshold T]
classify_site train --labels FILE --sites DIR [--out model.json]
classify_site eval  --labels FILE --sites DIR [--folds N] [--features all|signals|tokens] [--verbose] [--sweep]
`)+"\n")
	}
}
