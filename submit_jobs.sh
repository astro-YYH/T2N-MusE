#!/bin/bash

base="opt_submit_z"
i_start=0
i_end=32

for i in $(seq $i_start $i_end); do
  job_name="${base}$(printf "%02d" "$i")"
  echo "🔸 Submitting job: $job_name"
  sbatch "${job_name}.sh"
done

