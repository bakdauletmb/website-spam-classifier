# Spam classifier (Go). Run `make help` for targets.
GO      ?= go
BIN     := bin/classify_site
LABELS  ?= data/labels.txt
SITES   ?= data/sites
SITE    ?= example.com.html
FOLDS   ?= 5

.PHONY: help deps build test train eval eval-all classify clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

deps: ## Download Go module dependencies
	$(GO) mod download

build: deps ## Build the CLI into bin/classify_site (embeds model.json)
	$(GO) build -o $(BIN) .

test: ## Run unit tests
	$(GO) test ./...

train: deps ## Retrain model.json from data/labels.txt + data/sites, then rebuild
	$(GO) run . train --labels $(LABELS) --sites $(SITES) --out model.json
	$(GO) build -o $(BIN) .

eval: deps ## Stratified k-fold cross-validation with the default feature set
	$(GO) run . eval --labels $(LABELS) --sites $(SITES) --folds $(FOLDS) --sweep

eval-all: deps ## Cross-validate each feature family separately (ablation)
	@for m in all signals tokens; do \
	  echo "== features=$$m"; \
	  $(GO) run . eval --labels $(LABELS) --sites $(SITES) --folds $(FOLDS) --features $$m 2>/dev/null | grep overall; \
	done

classify: build ## Classify one site: make classify SITE=path/to/site
	./$(BIN) --input-web-site $(SITE) --explain

clean: ## Remove build artifacts
	rm -rf bin
