CC = gcc
AR = ar
CPPFLAGS +=	-I include/
CFLAGS += -g -Wall -O2
LDFLAGS += $(LIBS) -lz -lm -lpthread
BUILD_DIR = lib

# change the tool name to what you want
BINARY = quantrnn

STATICLIB = $(BUILD_DIR)/libquantrnn.a

OBJ = $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/decode_cpu.o \
	  $(BUILD_DIR)/quantrnn.o \

GPU_LIB =

# add more objects here if needed
VERSION = `git describe --tags`

# make asan=1 enables address sanitiser
ifdef asan
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make accel=1 enables the acceelerator (CUDA,OpenCL,FPGA etc if implemented)
ifdef cuda
	CUDA_ROOT ?= /usr/local/cuda
    CUDA_LIB ?= $(CUDA_ROOT)/lib64
    CUDA_OBJ += $(BUILD_DIR)/decode_cuda.o $(BUILD_DIR)/beam_search_cuda.o $(BUILD_DIR)/scan_cuda.o $(BUILD_DIR)/nn_cuda.o
    NVCC ?= $(CUDA_ROOT)/bin/nvcc
    CUDA_CFLAGS += -g -O2 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    GPU_LIB = $(BUILD_DIR)/cuda.a
    CPPFLAGS += -DHAVE_CUDA=1
else
	GPU_LIB = $(BUILD_DIR)/cpu_decoy.a
endif

ifdef bench
	CPPFLAGS += -DBENCH=1
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
	CFLAGS += -fopenmp
endif

.PHONY: clean distclean test

$(BINARY): $(BUILD_DIR)/main.o $(STATICLIB)
	$(CC) $(CFLAGS) $(BUILD_DIR)/main.o $(STATICLIB) $(LDFLAGS) $(CUDA_LDFLAGS) $(ROCM_LDFLAGS) -o $@

$(STATICLIB): $(OBJ) $(GPU_LIB)
	cp $(GPU_LIB) $@
	$(AR) rcs $@ $(OBJ)

$(BUILD_DIR)/main.o: src/main.c include/quantrnn/quantrnn.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.c include/quantrnn/quantrnn_error.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/quantrnn.o: src/quantrnn.c include/quantrnn/quantrnn.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# cpu decoy
$(BUILD_DIR)/cpu_decoy.a:
	rm -f $@
	$(AR) -r $@

# cuda
$(BUILD_DIR)/cuda.a: $(BUILD_DIR)/cuda_code.o $(CUDA_OBJ)
	$(AR) rcs $@ $^

$(BUILD_DIR)/cuda_code.o: $(CUDA_OBJ)
	$(NVCC) $(CUDA_CFLAGS) -dlink $^ -o $@

$(BUILD_DIR)/decode_cuda.o: src/decode_cuda.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -rdc=true -c $< -o $@

clean:
	rm -rf $(BINARY) $(BUILD_DIR)/*

# Delete all gitignored files (but not directories)
distclean: clean
	git clean -f -X
	rm -rf $(BINARY) $(BUILD_DIR)/* autom4te.cache

# make test with run a simple test
test: $(BINARY)
	./test/test.sh