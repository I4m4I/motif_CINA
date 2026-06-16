#!/usr/bin/env bash

set -euo pipefail

datasets=("smnist")
run_ids=(0 1 2 3 4 5 6 7 8 9)

# prefixes=(
# 	"MOP_E"
# 	"MOP"
# 	"AVE"
# 	"FRP"
# 	"FRP_E"
# )

# fre_lists=(
# 	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.349035 -1"
# 	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.249035 -1"
# 	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# 	"0.091003 0.287659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# 	"0.09003 0.387659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1"
# )

prefixes=(
	"2"
	"2E"
	"12"
	"12E"
	"Vanilla"
)

fre_lists=(
	"-1 0.25 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
	"-1 0.4 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.25 -1"
	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.4 -1"
	"-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1"
)

for dataset in "${datasets[@]}"; do
	for run_id in "${run_ids[@]}"; do
		for idx in "${!prefixes[@]}"; do
			prefix="${prefixes[$idx]}"
			fre="${fre_lists[$idx]}"
			echo "Running dataset=${dataset}, prefix=${prefix}, run_id=${run_id}"
			python -m classified.main train \
				--dataset "${dataset}" \
				--run-id "${run_id}" \
				--prefix "${prefix}" \
				--fre ${fre}
		done
	done
done

