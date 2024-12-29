#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

BASE_DIR=~/Desktop/Bubble-GPU

FLUID_MODEL=$1
PHASE_MODEL=$2
ID=$3

if [ -z "$FLUID_MODEL" ] || [ -z "$PHASE_MODEL" ] || [ -z "$ID" ]; then
    echo -e "${RED}Erro: Argumentos insuficientes. Uso: ./pipeline.sh <fluid_model> <phase_model> <id>${RESET}"
    exit 1
fi

MODEL_DIR=$BASE_DIR/bin/${FLUID_MODEL}_${PHASE_MODEL}
SIMULATION_DIR=${MODEL_DIR}/${ID}

# Cria o diretório do modelo e a subpasta específica para o ID
echo -e "${YELLOW}Preparando o diretório ${CYAN}${SIMULATION_DIR}${RESET}"
mkdir -p $SIMULATION_DIR
# Adiciona o arquivo .gitkeep no diretório vazio
touch ${SIMULATION_DIR}/.gitkeep
rm -rf ${SIMULATION_DIR}/*
FILES=$(ls -A ${SIMULATION_DIR} | grep -v '^\.gitkeep$')
if [ -n "$FILES" ]; then
    echo -e "${RED}Erro: O diretório ${CYAN}${SIMULATION_DIR}${RED} ainda contém arquivos!${RESET}"
    exit 1
else
    echo -e "${GREEN}Diretório limpo com sucesso.${RESET}"
fi

# Atualiza automaticamente a .gitignore
GITIGNORE_PATH=$BASE_DIR/.gitignore
SIM_RELATIVE_PATH="bin/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/*"
KEEP_RELATIVE_PATH="!bin/${FLUID_MODEL}_${PHASE_MODEL}/${ID}/.gitkeep"

# Adiciona as entradas ao .gitignore, se não estiverem presentes
if ! grep -Fxq "${SIM_RELATIVE_PATH}" "${GITIGNORE_PATH}"; then
    echo "${SIM_RELATIVE_PATH}" >> "${GITIGNORE_PATH}"
    echo "${KEEP_RELATIVE_PATH}" >> "${GITIGNORE_PATH}"
    echo -e "${GREEN}Entradas adicionadas à .gitignore:${RESET}"
    echo -e "  ${CYAN}${SIM_RELATIVE_PATH}${RESET}"
    echo -e "  ${CYAN}${KEEP_RELATIVE_PATH}${RESET}"
else
    echo -e "${YELLOW}Entradas já existem na .gitignore:${RESET}"
    echo -e "  ${CYAN}${SIM_RELATIVE_PATH}${RESET}"
    echo -e "  ${CYAN}${KEEP_RELATIVE_PATH}${RESET}"
fi

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
cd $BASE_DIR/src || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}sh compile.sh ${FLUID_MODEL} ${PHASE_MODEL} ${ID}${RESET}"
sh compile.sh ${FLUID_MODEL} ${PHASE_MODEL} ${ID} || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

echo -e "${YELLOW}Indo para ${CYAN}${MODEL_DIR}${RESET}"
cd ${MODEL_DIR} || { echo -e "${RED}Erro: Diretório ${CYAN}${MODEL_DIR}${RED} não encontrado!${RESET}"; exit 1; }

EXECUTABLE=${ID}sim_${FLUID_MODEL}_${PHASE_MODEL}_sm86
echo -e "${BLUE}Executando: ${CYAN}sudo ./${EXECUTABLE} ${FLUID_MODEL} ${PHASE_MODEL} ${ID}${RESET}"
sudo ./${EXECUTABLE} ${FLUID_MODEL} ${PHASE_MODEL} ${ID} || { echo -e "${RED}Erro na execução do simulador${RESET}"; exit 1; }

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/post${RESET}"
cd $BASE_DIR/post || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/post${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}./post.sh ${ID} ${FLUID_MODEL} ${PHASE_MODEL}${RESET}"
./post.sh ${ID} ${FLUID_MODEL} ${PHASE_MODEL} || { echo -e "${RED}Erro na execução do script post.sh${RESET}"; exit 1; }

echo -e "${GREEN}Processo concluído com sucesso!${RESET}"
