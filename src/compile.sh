CC=86

nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
    *.cu \
    -lcudadevrt -lcurand -o ./../bin/${2}sim_${1}_sm${CC}