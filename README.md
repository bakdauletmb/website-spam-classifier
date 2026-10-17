# Web Site Spam Classifier

A command-line tool that decides whether a scraped web page is spam, and
explains why. It combines SpamAssassin-style hand-written signals
(parked-domain templates, redirect stubs, keyword stuffing, link-farm shape,
missing site structure) with a bag-of-words model, and learns the weight of
every signal with L2-regularized logistic regression written in plain Go.
The trained model is embedded in the binary, so classifying a page needs no
extra files.

```
$ ./bin/classify_site --input-web-site example.com.html
URL 'example.com' is not spam.
```

## Quick start

Requires Go 1.24. The only third-party dependency is `golang.org/x/net`
(HTML5 parser and charset sniffing), pinned in `go.mod`.

```bash
cd go
make build                                   # -> bin/classify_site
curl -sL https://example.com -o example.com.html
./bin/classify_site --input-web-site example.com.html
./bin/classify_site --input-web-site example.com.html --explain
```

Without `make`: `go build -o bin/classify_site .` builds and `go test ./...`
runs the unit tests.

### Input format

`--input-web-site` takes a path to a scraped site: either a single HTML file
named after the host (`example.com.html` or just `example.com`) or a
directory that contains `index.html`. The host name is read from the path
and feeds the domain-shape features.

## Explaining a verdict

`--explain` prints the spam probability and the features that moved the
decision most. For the page fetched above:

```
URL 'example.com' is not spam.
  spam probability: 0.441 (threshold 0.50, 44 active features)
  strongest evidence:
     +0.87  sig:spam_vocab_density
     +0.80  sig:images<=0
     -0.63  sig:hidden<=0
     -0.62  sig:ad_hits<=0
     -0.61  sig:iframes<=0
     -0.61  sig:domain_len<=10
     +0.59  sig:has_viewport
     +0.53  w:domain
     +0.52  sig:no_meta_desc
     +0.52  sig:scripts<=1
     +0.44  ti:domain
     -0.41  sig:lexical_diversity (value 0.88)
```

A one-paragraph page with no images, no links and no description is thin,
which is also what parked domains look like, so the score sits near the
middle. The absence of hidden text, ads and iframes keeps it on the
legitimate side. `--json` emits the same information as JSON.

## How it works

1. **Parse.** The page is decoded with charset sniffing and parsed with
   `golang.org/x/net/html`. Visible text is separated from hidden text
   (`display:none`, `visibility:hidden`, zero font size, off-screen
   positioning, `<noscript>`), and anchors, scripts, iframes, images, forms,
   headings and meta tags are collected.
2. **Signals.** About 170 human-readable features in `internal/features`.
   Counts are bucketed into indicators (`sig:images<=0`) and ratios are
   clipped to [0,1], so every value is bounded and a weight reads as
   "evidence for spam".

   | Family | Examples |
   |---|---|
   | Templates | parked-domain page, error page, JS redirect stub, script-only or iframe-only stub |
   | Stuffing | lexical diversity, top-word share, spam-vocabulary hits and density, repeated keywords |
   | Link profile | external-link ratio, empty anchors, long anchors, distinct external hosts, affiliate links |
   | Ads and stubs | ad-network hits, ads with thin content, more hidden than visible text |
   | Site structure | copyright line, navigation markers, forms and inputs, viewport, Open Graph, RSS |
   | Domain shape | TLD, length, hyphens, digits, many tokens, spam vocabulary in the name |

3. **Tokens.** A bag of words with a source prefix: `ti:` title, `me:` meta
   description, `w:` body, `an:` anchor text, `hd:` headings, `dom:` domain
   parts, `lh:` link hosts, `ih:` iframe hosts, `sh:` script hosts.
4. **Model.** Logistic regression trained by SGD with lazy L2
   regularization, a decaying learning rate and minimum document
   frequency pruning, implemented in `internal/model` with no dependencies.
   Training is deterministic. The model is a JSON map of feature name to
   weight, and the explanation is simply the largest `weight × value`
   terms.

## Results

The shipped model was trained on a hand-labeled set of 1,172 home pages
(407 spam, 765 legitimate) collected around 2013. The data set is not
included in this repository. Stratified 5-fold cross-validation on it:

| Features                   | Accuracy | Precision | Recall | F1    | False-positive rate |
|----------------------------|---------:|----------:|-------:|------:|--------------------:|
| tokens only                | 0.900    | 0.922     | 0.779  | 0.844 | 3.5%                |
| signals only               | 0.908    | 0.877     | 0.855  | 0.866 | 6.4%                |
| signals + tokens (shipped) | 0.921    | 0.918     | 0.848  | 0.881 | 4.1%                |

Raising `--threshold` to 0.8 lowers the false-positive rate to 1.0% at the
cost of recall (0.789). Classification takes about 5 ms per page on a
laptop. The pages are a decade old, so a freshly fetched modern page is a
sanity check rather than a measurement.

## Training your own model

Put a label file and a directory of scraped pages under `go/data/`, which
is git-ignored:

```
go/data/labels.txt      one "<host>\t<0|1>" per line (1 = spam); spam/ham and true/false also work
go/data/sites/<host>    the scraped HTML, named after the host (".html" suffix is fine)
```

Then, from `go/`:

```bash
make eval        # stratified k-fold cross-validation plus a threshold sweep
make eval-all    # same, once per feature family (ablation)
make train       # rewrite model.json and rebuild the binary with it embedded
go run . eval --verbose          # also lists every misclassified site
go run . train --help            # epochs, learning rate, L2, min-df, pos-weight
```

## Command-line reference

```
classify_site --input-web-site PATH [--explain] [--json] [--model FILE] [--threshold T]
classify_site train --labels FILE --sites DIR [--out model.json] [--epochs N] [--lr F] [--l2 F] [--min-df N]
classify_site eval  --labels FILE --sites DIR [--folds N] [--features all|signals|tokens] [--verbose] [--sweep]
```

Exit codes: 0 when a verdict was printed (spam or not), 1 when the input is
missing or unreadable, 2 for invalid flags.

## Layout

```
go/
  main.go                 CLI (classify / train / eval)
  model.json              trained weights, embedded at build time
  internal/page           HTML loading, charset handling, text extraction
  internal/features       signal rules + token features -> sparse vector
  internal/model          logistic regression (train, score, explain, JSON)
  internal/dataset        label file parsing, stratified folds
  internal/eval           confusion-matrix metrics
```

## Troubleshooting

If `go build` tries to download a newer toolchain, check that `go version`
reports 1.24.x and that `go.mod` has not been bumped by a `go get` of a
newer dependency. The module pins `golang.org/x/net` to a release that
builds with Go 1.24.
