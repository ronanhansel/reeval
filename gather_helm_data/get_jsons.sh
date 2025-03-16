#!/bin/bash
set -e

mkdir -p ./helm_jsons

# Define benchmarks and their GCS paths.
declare -A gcs_paths=(
  [lite]="gs://crfm-helm-public/lite/benchmark_output"
  [classic]="gs://crfm-helm-public/benchmark_output"
  [mmlu]="gs://crfm-helm-public/mmlu/benchmark_output"
  [air-bench]="gs://crfm-helm-public/air-bench/benchmark_output"
  [cleva]="gs://crfm-helm-public/cleva/benchmark_output"
  [decodingtrust]="gs://crfm-helm-public/decodingtrust/benchmark_output"
  [heim]="gs://crfm-helm-public/heim/benchmark_output"
  [instruct]="gs://crfm-helm-public/instruct/benchmark_output"
  [image2structure]="gs://crfm-helm-public/image2structure/benchmark_output"
  [safety]="gs://crfm-helm-public/safety/benchmark_output"
  [thaiexam]="gs://crfm-helm-public/thaiexam/benchmark_output"
  [vhelm]="gs://crfm-helm-public/vhelm/benchmark_output"
)

# Define release versions for each benchmark (if applicable).
declare -A releases=(
  [lite]="v1.12.0 v1.13.0"
  [classic]="v0.4.0"
  [mmlu]="v1.12.0 v1.13.0"
  [air-bench]="v1.3.0 v1.4.0"
  [cleva]="v1.0.0"
  [decodingtrust]="v0.1.0"
  [heim]="v1.1.0"
  [instruct]="v1.0.0"
  [safety]="v1.0.0"
  [thaiexam]="v1.1.0"
)

# Define suite run versions for each benchmark.
declare -A suites=(
  [lite]="v1.{0..13}.0"
  [classic]="v0.3.0 v0.4.0"
  [mmlu]="v1.{0..13}.0"
  [air-bench]="v1.{0..4}.0"
  [cleva]="v1.0.0"
  [decodingtrust]="v0.1.0"
  [heim]="v1.1.0"
  [instruct]="v1.0.0"
  [image2structure]="v1.0.2"
  [safety]="v1.0.0"
  [thaiexam]="v1.1.0 v1.0.0"
  [vhelm]="v2.1.0"
)

for benchmark in "${!gcs_paths[@]}"; do
  echo "Syncing benchmark: $benchmark"
  local_path="./helm_jsons/$benchmark"
  mkdir -p "$local_path"
  gcs_path="${gcs_paths[$benchmark]}"

  # Sync releases if defined.
  if [ -n "${releases[$benchmark]}" ]; then
    for ver in ${releases[$benchmark]}; do
      echo "  - Release: $ver"
      mkdir -p "$local_path/releases/$ver"
      gcloud storage rsync -r "$gcs_path/releases/$ver" "$local_path/releases/$ver"
    done
  fi

  # Sync suite runs if defined.
  if [ -n "${suites[$benchmark]}" ]; then
    # Expand brace notation if present.
    for ver in $(eval echo ${suites[$benchmark]}); do
      echo "  - Suite run: $ver"
      mkdir -p "$local_path/runs/$ver"
      gcloud storage rsync -r "$gcs_path/runs/$ver" "$local_path/runs/$ver"
    done
  fi
done
