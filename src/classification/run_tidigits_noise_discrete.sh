#!/usr/bin/env bash

set -euo pipefail

datasets=("tidigits")
run_ids=(0 1 2 3 4 5 6 7 8 9)

prefixes=(
	"2"
	"2E"
	"12"
	"12E"
	"Vanilla"
)

# prefixes=(
# 	"Vanilla"
# )

max_var=1.0
steps=26
discrete_levels=32
device="auto"

for dataset in "${datasets[@]}"; do
	for run_id in "${run_ids[@]}"; do
		for prefix in "${prefixes[@]}"; do
			echo "Running noise eval: dataset=${dataset}, prefix=${prefix}, run_id=${run_id}, discrete_levels=${discrete_levels}"
			python -m classified.main noise \
				--dataset "${dataset}" \
				--run-id "${run_id}" \
				--prefix "${prefix}" \
				--max-var "${max_var}" \
				--steps "${steps}" \
				--device "${device}" \
				--discrete \
				--discrete-levels "${discrete_levels}"
		done
	done
done
