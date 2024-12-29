#!/bin/bash

CC=86

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3

OUTPUT_DIR=~/Desktop/Bubble-GPU/bin/${FLUID_MODEL}_${PHASE_MODEL}

mkdir -p ${OUTPUT_DIR}
nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
    ~/Desktop/Bubble-GPU/src/*.cu \
    -lcudadevrt -lcurand -D${FLUID_MODEL} -D${PHASE_MODEL} -o ${OUTPUT_DIR}/${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm${CC}
