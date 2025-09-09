#################################
# RISCV Toolchain
#################################

PREFIX = riscv64-unknown-elf-

GCC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

sysroot = $(shell $(GCC) -print-sysroot)
# GCC = clang
# sysroot = /opt/rv64gcv_lp64d_enable_llvm/riscv64-unknown-elf
#################################
# Flags
#################################

# SoC Settings
ARCH = rv64imafdcv
ABI = lp64d
ARCHFLAGS = -march=$(ARCH) -mabi=$(ABI)

CFLAGS  = -std=gnu99 -g -fno-common -fno-builtin-printf -fpermissive -w -O1
CFLAGS += $(ARCHFLAGS)
LDFLAGS = -static -lm

# include libgloss.mk

libgloss_specs := htif_nano.specs
CFLAGS += -specs=$(libgloss_specs)
LDFLAGS += -specs=$(libgloss_specs)

PROGRAMS = ibert-base softmax_cute transformer_cute ibert_2_notcm reduce ibert_2_seg ibert_2_vecdiv ibert_2_softpipe ibert_6_softpipe ibert_4_res

NUMBERS := $(shell seq 1 6)

NEWPROGRAMS :=
$(foreach i, $(NUMBERS), \
    $(eval X = $(shell echo $$(($i)))) \
    $(eval NEWPROGRAMS += ibert_$(X)) \
)

PROGRAMS += $(NEWPROGRAMS)

IBERTBASE :=
$(foreach i, $(NUMBERS), \
    $(eval X = $(shell echo $$(($i)))) \
    $(eval IBERTBASE += ibert-base-$(X)) \
)

PROGRAMS += $(IBERTBASE)

.DEFAULT_GOAL := default


#################################
# Build
#################################


%.o: %.S
	$(GCC) $(CFLAGS) -D__ASSEMBLY__=1 -c $< -o $@

%.o: %.c
	$(GCC) $(CFLAGS) -c $< -o $@

%.riscv: %.o $(libgloss)
	$(GCC) $(LDFLAGS) $< -o $@

%.dump: %.riscv
	$(OBJDUMP) -D -S $< > $@

# gcc get_test.c -o get_value
get_matrix_value: get_matrix_test.c
	$(CC) -o $@ $^

get_conv_value: get_conv_test.c
	$(CC) -o $@ $^

#################################
# Recipes
#################################

.PHONY: clean
clean:
	rm -f *.riscv *.o *.dump *.S
	$(if $(libgloss),rm -rf $(libgloss_builddir)/)

.PHONY: default
default: $(addsuffix .riscv, $(PROGRAMS))

.PHONY: dumps
dumps: $(addsuffix .dump, $(PROGRAMS))
