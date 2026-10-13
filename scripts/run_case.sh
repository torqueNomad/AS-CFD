#!/bin/bash
# Run simpleFoam on a meshed, decomposed case (4 MPI ranks). Usage: run_case.sh <case_dir>
source /usr/lib/openfoam/openfoam2512/etc/bashrc
set -e
export OMPI_MCA_btl_vader_single_copy_mechanism=none
cd "$1"
mpirun --allow-run-as-root --oversubscribe -np 4 simpleFoam -parallel > log.simpleFoam 2>&1
