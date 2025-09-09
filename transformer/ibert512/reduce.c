#include <stdio.h>
#include <string.h>
#include <stdbool.h>
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"


// #include <stdint.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
// #define XCUSTOM_ACC 3
// #define DIM 16
// #define ADDR_LEN 32
// #define BANK_NUM 4
// #define BANK_ROWS 4096
// #define ACC_ROWS 1024
// #define MAX_BYTES 64
// #define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
// #define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))

static uint64_t read_cycles() {
    
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

}

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

#define HAS_MVIN_SCALE
typedef float scale_t;
typedef uint32_t scale_t_bits;

typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;

typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;


#define MVIN_SCALE_IDENTITY 1.0

#define ACC_SCALE_IDENTITY 1.0

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ((shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))



int main (int argc, char * argv[]) {

    acc_t sum_exp_sum = 0;
    acc_t max_q[16] __attribute__((aligned(64))) = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __asm__ volatile (
        "mv a6, %[max_q]                \n"

        "li t1, 16                          \n" //

        "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
        "vle32.v v1, (a6)           \n" // 加载 max_q 的 8 个元素到 v1
        "li t0, 0                           \n" //


        "vmv.v.x    v0, x0          \n"
        "vredsum.vs v0, v1, v0      \n"
        "vmv.x.s    %[sum_exp_sum], v0          \n"

        : [sum_exp_sum] "=r" (sum_exp_sum)
        : [max_q] "r" (max_q)  // 输入约束
        : "t0", "t1", "t2", "t3", "t4", "t5", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "memory" // 破坏描述符
    );
    printf("sum_exp_sum = %d\n", sum_exp_sum);
}

