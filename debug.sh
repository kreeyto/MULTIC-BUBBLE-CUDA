#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

BASE_DIR=$(dirname "$(readlink -f "$0")")

# Definindo valores padrão
FLUID_MODEL=${1:-FD3Q19}
PHASE_MODEL=${2:-PD3Q15}
ID=${3:-000}

echo -e "${YELLOW}Indo para ${CYAN}$BASE_DIR/src${RESET}"
cd $BASE_DIR/src || { echo -e "${RED}Erro: Diretório ${CYAN}$BASE_DIR/src${RED} não encontrado!${RESET}"; exit 1; }

# Chamando o compile.sh com DEBUG_MODE=1
echo -e "${BLUE}Executando: ${CYAN}sh compile.sh${RESET}"
sh compile.sh ${FLUID_MODEL} ${PHASE_MODEL} ${ID} 0 1 || { echo -e "${RED}Erro na execução do script compile.sh${RESET}"; exit 1; }

# Verificando o executável debugexe
EXECUTABLE="${BASE_DIR}/debugexe"

if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Erro: Executável debugexe não encontrado em ${CYAN}${EXECUTABLE}${RESET}"
    exit 1
fi

# Executando o debugexe
echo -e "${BLUE}Executando: ${CYAN}${EXECUTABLE}${RESET}"
"${EXECUTABLE}" "${FLUID_MODEL}" "${PHASE_MODEL}" "${ID}" 0 1 || {
    echo -e "${RED}Erro na execução do executável debugexe${RESET}"
    exit 1
}

echo -e "${GREEN}Execução concluída com sucesso!${RESET}"
