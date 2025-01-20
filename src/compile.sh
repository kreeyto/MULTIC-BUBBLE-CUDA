#!/bin/bash

CC=86

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3

BASE_DIR=$(dirname "$0")/..
SRC_DIR=${BASE_DIR}/src
OUTPUT_DIR=${BASE_DIR}/bin/${FLUID_MODEL}_${PHASE_MODEL}

mkdir -p ${OUTPUT_DIR}

nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
    ${SRC_DIR}/*.cu \
    -lcudadevrt -lcurand -D${FLUID_MODEL} -D${PHASE_MODEL} -o ${OUTPUT_DIR}/${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm${CC}
