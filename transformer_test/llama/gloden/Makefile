# Compiler and flags
CC = gcc
# -fpermissive
CFLAGS = -fpermissive -O1 -w -g
LDFLAGS = -lm

# Target executable
GLODEN = llama_x86_golden llama_x86_golden_workload
TEST = llama


# Source file
# SRC = llama_x86_golden.c

# Rule to create the executable
.PHONY: default
default: $(addsuffix .out, $(TEST))
gloden: $(addsuffix .out, $(GLODEN))

%.out: %.c
	$(CC) $(CFLAGS)  $< -o $@ $(LDFLAGS)



# Clean up the generated files
clean:
	rm -f $(addsuffix .out, $(GLODEN) $(TEST))

.PHONY: all clean