#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

BASE_DIR=$(dirname "$(readlink -f "$0")")

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3
SAVE_BINARY=$4

if [ -z "$FLUID_MODEL" ] || [ -z "$PHASE_MODEL" ] || [ -z "$ID" ] || [ -z "$SAVE_BINARY" ]; then
    echo -e "${RED}Erro: Argumentos insuficientes. Uso: ./pipeline.sh <fluid_model> <phase_model> <id> <save_binary>${RESET}"
    exit 1
fi

if [ "$SAVE_BINARY" -eq 0 ]; then
    MODEL_DIR=$BASE_DIR/matlabFiles/${FLUID_MODEL}_${PHASE_MODEL}
else
    MODEL_DIR=$BASE_DIR/bin/${FLUID_MODEL}_${PHASE_MODEL}
fi

SIMULATION_DIR=${MODEL_DIR}/${ID}
echo -e "${YELLOW}Preparando os diretórios ${CYAN}${SIMULATION_DIR}${RESET}"
mkdir -p ${SIMULATION_DIR}

echo -e "${YELLOW}Limpando o diretório ${CYAN}${SIMULATION_DIR}${RESET}"
find "${SIMULATION_DIR}" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

FILES=$(ls -A ${SIMULATION_DIR} | grep -v '^\.gitkeep$')
if [ -n "$FILES" ]; then
    echo -e "${RED}Erro: O diretório ${CYAN}${SIMULATION_DIR}${RED} ainda contém arquivos!${RESET}"
    exit 1
else
    echo -e "${GREEN}Diretório limpo com sucesso.${RESET}"
fi

GITIGNORE_PATH=$BASE_DIR/.gitignore
SIM_RELATIVE_PATH="bin/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/*"
KEEP_RELATIVE_PATH="!bin/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/.gitkeep"
MATLAB_RELATIVE_PATH="matlabFiles/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/*"
KEEP_MATLAB_PATH="!matlabFiles/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/.gitkeep"

if [ "$SAVE_BINARY" -ne 0 ] && ! grep -Fxq "${SIM_RELATIVE_PATH}" "${GITIGNORE_PATH}"; then
    echo "${SIM_RELATIVE_PATH}" >> "${GITIGNORE_PATH}"
    echo "${KEEP_RELATIVE_PATH}" >> "${GITIGNORE_PATH}"
    echo -e "${GREEN}Entradas adicionadas à .gitignore (bin):${RESET}"
fi

if [ "$SAVE_BINARY" -eq 0 ] && ! grep -Fxq "${MATLAB_RELATIVE_PATH}" "${GITIGNORE_PATH}"; then
    echo "${MATLAB_RELATIVE_PATH}" >> "${GITIGNORE_PATH}"
    echo "${KEEP_MATLAB_PATH}" >> "${GITIGNORE_PATH}"
    echo -e "${GREEN}Entradas adicionadas à .gitignore (matlabFiles):${RESET}"
fi

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
cd $BASE_DIR/src || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}sh compile.sh ${FLUID_MODEL} ${PHASE_MODEL} ${ID} ${SAVE_BINARY}${RESET}"
sh compile.sh ${FLUID_MODEL} ${PHASE_MODEL} ${ID} ${SAVE_BINARY} || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

EXECUTABLE=$(realpath "${MODEL_DIR}/${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm86")

if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Erro: Executável não encontrado em ${CYAN}${EXECUTABLE}${RESET}"
    exit 1
fi

echo -e "${BLUE}Executando: ${CYAN}sudo ${EXECUTABLE} ${FLUID_MODEL} ${PHASE_MODEL} ${ID} ${SAVE_BINARY}${RESET}"
sudo "${EXECUTABLE}" "${FLUID_MODEL}" "${PHASE_MODEL}" "${ID}" "${SAVE_BINARY}" || {
    echo -e "${RED}Erro na execução do simulador${RESET}"
    exit 1
}

if [ "$SAVE_BINARY" -eq 0 ]; then
    echo -e "${GREEN}Execução concluída com sucesso. Pipeline encerrada devido a save_binary=0.${RESET}"
    exit 0
fi

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/post${RESET}"
cd $BASE_DIR/post || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/post${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}./post.sh ${ID} ${FLUID_MODEL} ${PHASE_MODEL}${RESET}"
./post.sh ${ID} ${FLUID_MODEL} ${PHASE_MODEL} || { echo -e "${RED}Erro na execução do script post.sh${RESET}"; exit 1; }

echo -e "${GREEN}Processo concluído com sucesso!${RESET}"
