#!/bin/bash

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'


BASE_DIR=~/Desktop/Bubble-GPU

echo -e "${YELLOW}Limpando o diretório ${CYAN}$BASE_DIR/bin/simulation/000${RESET}"
rm -rf $BASE_DIR/bin/simulation/000/*
FILES=$(ls -A $BASE_DIR/bin/simulation/000 | grep -v '^\.gitkeep$')
if [ -n "$FILES" ]; then
    echo -e "${RED}Erro: O diretório ${CYAN}$BASE_DIR/bin/simulation/000${RED} ainda contém arquivos!${RESET}"
    exit 1
else
    echo -e "${GREEN}Diretório limpo com sucesso.${RESET}"
fi

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
cd $BASE_DIR/src || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}sh compile.sh D3Q19 000${RESET}"
sh compile.sh D3Q19 000 || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/bin${RESET}"
cd $BASE_DIR/bin || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/bin${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}sudo ./000sim_D3Q19_sm86${RESET}"
sudo ./000sim_D3Q19_sm86 || { echo -e "${RED}Erro na execução do simulador${RESET}"; exit 1; }

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/post${RESET}"
cd $BASE_DIR/post || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/post${RED} não encontrado!${RESET}"; exit 1; }

echo -e "${BLUE}Executando: ${CYAN}./post.sh${RESET}"
./post.sh || { echo -e "${RED}Erro na execução do script post.sh${RESET}"; exit 1; }

echo -e "${GREEN}Processo concluído com sucesso!${RESET}"

