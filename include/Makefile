CC = nvcc
SRC = main.cu kernels.cu utils.cu constants.cu
OBJ = $(SRC:.cu=.o)
TARGET = simulation

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -o $@ $^

%.o: %.cu
	$(CC) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
