#!/usr/bin/env bash
set -euo pipefail

# .gh-pages-push is prepared once by the timeline workflow. The complete
# ASV result tree also carries the persistent rerun checkpoint.
asv publish
find .gh-pages-push -mindepth 1 -maxdepth 1 \
  ! -name .git ! -name experiments -exec rm -rf {} +
cp -a .asv/html/. .gh-pages-push/
mkdir -p .gh-pages-push/results
cp -a .asv/results/. .gh-pages-push/results/
python scripts/make_dashboard.py .gh-pages-push .gh-pages-push/dashboard.html
if [ -d .gh-pages-push/experiments ]; then
  python scripts/make_experiment_dashboard.py .gh-pages-push .gh-pages-push/experiments.html
fi

git -C .gh-pages-push add -A
if ! git -C .gh-pages-push diff --cached --quiet; then
  git -C .gh-pages-push commit -q -m "Publish timeline checkpoint ${BENCHMARK_SHA:-rerun}"
fi
git -C .gh-pages-push pull --rebase --autostash -q origin gh-pages
git -C .gh-pages-push push -q origin gh-pages
