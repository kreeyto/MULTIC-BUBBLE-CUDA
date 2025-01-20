#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Uso: ./post.sh <ID> <fluid_model> <phase_model>"
    exit 1
fi

ID=$1   
FLUID_MODEL=$2
PHASE_MODEL=$3

python3 exampleVTK.py "$ID" "$FLUID_MODEL" "$PHASE_MODEL"
