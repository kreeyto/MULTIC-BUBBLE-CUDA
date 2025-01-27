#!/bin/bash

CC=86

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3
SAVE_BINARY=$4 # Default para 0 se SAVE_BINARY não for fornecido
DEBUG_MODE=$5  # Novo argumento para indicar o modo debug (0 ou 1)

BASE_DIR=$(dirname "$0")/..
SRC_DIR=${BASE_DIR}/src

if [ "$DEBUG_MODE" -eq 1 ]; then
    OUTPUT_DIR=${BASE_DIR}  # Para debug, o executável vai diretamente no BASE_DIR
    EXECUTABLE_NAME="debugexe"
else
    if [ "$SAVE_BINARY" -eq 0 ]; then
        OUTPUT_DIR=${BASE_DIR}/matlabFiles/${FLUID_MODEL}_${PHASE_MODEL}
    else
        OUTPUT_DIR=${BASE_DIR}/bin/${FLUID_MODEL}_${PHASE_MODEL}
    fi
    EXECUTABLE_NAME="${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm${CC}"
fi

mkdir -p ${OUTPUT_DIR}

echo "Compilando para ${OUTPUT_DIR}/${EXECUTABLE_NAME}..."

nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --disable-warnings -O3 --restrict \
    ${SRC_DIR}/*.cu \
    -lcudadevrt -lcurand -D${FLUID_MODEL} -D${PHASE_MODEL} \
    -o ${OUTPUT_DIR}/${EXECUTABLE_NAME}

if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Erro na compilação!"
    exit 1
fi
