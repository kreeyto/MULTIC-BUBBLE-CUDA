from dataSave import *
from fileTreat import *
import math
import sys

if len(sys.argv) < 2:
    print("Uso: python3 exampleVTK.py <ID>")
    sys.exit(1)

# Argumentos recebidos
simulation_id = sys.argv[1]
fluid_model = sys.argv[2]
phase_model = sys.argv[3]

# Caminho dinâmico baseado nos modelos e ID
path = f"./../bin/{fluid_model}_{phase_model}/{simulation_id}/"

# Verifica se o caminho existe
if not os.path.exists(path):
    print(f"Erro: O caminho {path} não existe.")
    sys.exit(1)

# Obtém os dados macroscópicos na pasta
macrSteps = getMacrSteps(path)
info = getSimInfo(path)

# for all steps saved
for step in macrSteps:
    macr = getMacrsFromStep(step,path)
    # Save macroscopics to VTK format
    print("Processing step", step)
    saveVTK3D(macr, path, info['ID'] + "macr" + str(step).zfill(6), points=True)