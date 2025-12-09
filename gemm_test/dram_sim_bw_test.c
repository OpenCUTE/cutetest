#include <stdio.h>
// #include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>

static uint64_t read_cycles() {
    
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

}

volatile char ram[8192 * 8192]__attribute__((aligned(64))) = {0};

int main(void)
{

    for(uint64_t stride = 0; stride < 8192; stride += 256) {
        uint64_t start = read_cycles();
        for(uint64_t i = 0; i < 8192; i+=32) {
            char tmp = ram[i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];
            tmp = ram[++i * stride];

        }
        uint64_t end = read_cycles();
        printf("stride %lu load cycles: %lu \n",stride, end - start);
    }

    return 0;
}
