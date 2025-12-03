#include <stdio.h>
#include <string.h>
#include <stdbool.h>
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

// #include <stdint.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "cuteMarcoinstHelper.h"
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

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#define MVIN_SCALE_IDENTITY 1.0

#define ACC_SCALE_IDENTITY 1.0

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ((shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#ifdef __cplusplus
#define SAME_TYPE(x) decltype(x)
#else
#define SAME_TYPE(x) typeof(x)
#endif

#define ROUND_NEAR_EVEN(x) \
    ({ const SAME_TYPE(x) x_ = (x); \
         const long long i = x_; \
         const long long next = x_ < 0 ? x_ - 1 : x_ + 1; \
         SAME_TYPE(x) rem = x_ - i; \
         rem = rem < 0 ? -rem : rem; \
         SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( \
                     i % 2 == 0 ? i : next)); \
         result; })

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT_BITS(x, shift) \
((shift) > 0 ? (((x) >> (shift)) + \
    (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
         ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#define ACC_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})

#define MVIN_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})

#define MVIN_SCALE_ACC(x, scale) (x)

#define ACC_SCALE_T_IS_FLOAT
#define ACC_SCALE_EXP_BITS 8
#define ACC_SCALE_SIG_BITS 24

#define ACC_READ_SMALL_WIDTH
#define ACC_READ_FULL_WIDTH

#define HAS_FIRST_LAYER_OPTIMIZATIONS

#endif // GEMMINI_PARAMS_H

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#include <math.h>

char *activation_name(int act) {
  switch (act) {
    case NO_ACTIVATION:
      return "NO_ACTIVATION";
    case RELU:
      return "RELU";
    case LAYERNORM:
      return "LAYERNORM";
    case IGELU:
      return "IGELU";
    case SOFTMAX:
      return "SOFTMAX";
    default:
      return "UNKNOWN";
  }
}

#define gemmini_fence() 

#define GEMMINI_ACC_SCALE(x, scale) (x)

#define GEMMINI_SCALE(x, scale) (x)

// static int tanh(int n){
//     return n/2.3333;
// }

static acc_t int_sqrt(acc_t n) {
  if (n <= 0) return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1) {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev) return x_prev;
    x_prev = x_next;
  };
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  return x;
}


enum tiled_matmul_type_t {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}

static void tiled_resadd_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

	  resadd_cpu(I, J, A_scale, B_scale, C_scale,
		A, B, C, relu);
}
// 512位对齐的数组acc_t result[64][64]
void * CUTE_result[2] = {(void *) (0x70200000), (void *) (0x70200000 + 64 * 3072 * 4)};//double buffer
int CUTE_result_index = 0;

elem_t * res_input, * res_output;

void softmax_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{
    // printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int) (0.693147 / scale)==0?1:(int) (0.693147 / scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / scale;
    const acc_t qc = c / (a*scale*scale);
    // printf("qln2:%d\n", qln2);
    // printf("qln2_inv:%d\n", qln2_inv);
    // printf("qb:%d\n", qb);
    // printf("qc:%d\n", qc);
    int j_itermax = dim_j / 16;
    int j_itermax_m4 = dim_j / 64;
    int j_itermax_store = dim_j / 64;
    int stride_in = 3072 * 4;

    // ----- startup -----
    size_t i_pipe0 = 0, i_pipe1;
    acc_t *input_slice_pipe0 = &input[i_pipe0][0], 
           *input_slice_pipe1;
    acc_t max_q_max = -2147483648;
    acc_t sum_exp_sum = 0;
    scale_t factor_pipe0, factor_pipe1;

    __asm__ volatile (
        "mv a3, %[input_slice_pipe0]              \n" // a3 = input 起始地址
        "mv a5, %[j_itermax_m4]                \n"

        "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
        "li t1, -2147483648                 \n" // max_q[16] __attribute__((aligned(64))) = {-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648}
        "vmv.v.x v4, t1                     \n" // 将 v1 寄存器的所有元素初始化为 -2147483648                     



        "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

        "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
        "vmax.vv v4, v4, v0       \n"   // v1 = max(v1, 0)

        "addi a3, a3, 256            \n"  // input的下一个起始地址
        "addi a5, a5, -1            \n" // 列计数器 a5--
        "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

        "vsetvli t0, zero, e32, m1  \n"
        "vmax.vv v4, v4, v5       \n"   // v1 = max(v1, 0)
        "vmax.vv v4, v4, v6       \n"   // v1 = max(v1, 0)
        "vmax.vv v4, v4, v7       \n"   // v1 = max(v1, 0)
        "vmv.v.x v0, t1             \n"
        "vredmax.vs v0, v4, v0      \n"
        "vmv.x.s %[max_q_max], v0          \n"

        : [max_q_max] "=r" (max_q_max)
        : [input_slice_pipe0] "r" (input_slice_pipe0), [j_itermax_m4] "r" (j_itermax_m4)  // 输入约束
        : "t0", "t1", "a3", "a4", "a5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory" // 破坏描述符
    );

    __asm__ volatile (
        "mv a3, %[input_slice_pipe0]              \n" // a3 = input 起始地址
        "mv a5, %[j_itermax_m4]                \n"
        "mv a6, %[max_q_max]                \n"
        "mv t2, %[qln2_inv]                 \n"
        "mv t3, %[qln2]                     \n"
        "mv t4, %[qb]                       \n"
        "mv t5, %[qc]                       \n"

        "li t1, 16                          \n" //

        "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
        "li t0, 0                           \n" //
        "vmv.v.x v4, t0           \n" // 加载 sum_exp 的 8 个元素到 v1

        "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

        "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
        "vsub.vx v0, v0, a6         \n" // q = input[i][j*8+k] - max_q_max;
        "vrsub.vi v8, v0, 0         \n" // -q
        "vmul.vx v8, v8, t2         \n" // -q * qln2_inv
        "vsra.vx v8, v8, t1         \n" // z = (acc_t) (-q * qln2_inv) >> 16;
        "vmul.vx v12, v8, t3         \n" // z * qln2
        "vadd.vv v0, v0, v12         \n" // qp = q + z * qln2;
        "vadd.vx v0, v0, t4         \n" // qp + qb
        "vmul.vv v0, v0, v0         \n" // (qp + qb)*(qp + qb)
        "vadd.vx v0, v0, t5         \n" // q_exp = (qp + qb)*(qp + qb) + qc;
        "vsra.vv v0, v0, v8         \n" // input[i][j*8+k] = q_exp >> z;
        "vadd.vv v4, v4, v0         \n" // sum_exp[k] += input[i][j*8+k];
        "vse32.v v0, (a3)           \n" // input[i][j*8+k] = q_exp >> z;

        "addi a3, a3, 256            \n"  // input的下一个起始地址
        "addi a5, a5, -1            \n" // 列计数器 a5--
        "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

        "vsetvli t0, zero, e32, m1  \n"
        "vadd.vv v4, v4, v5         \n"
        "vadd.vv v4, v4, v6         \n"
        "vadd.vv v4, v4, v7         \n"
        "vmv.v.x    v0, x0          \n"
        "vredsum.vs v0, v4, v0      \n"
        "vmv.x.s    %[sum_exp_sum], v0          \n"

        : [sum_exp_sum] "=r" (sum_exp_sum)
        : [input_slice_pipe0] "r" (input_slice_pipe0), [j_itermax_m4] "r" (j_itermax_m4), [max_q_max] "r" (max_q_max), [qln2_inv] "r" (qln2_inv), [qln2] "r" (qln2), [qb] "r" (qb), [qc] "r" (qc)  // 输入约束
        : "t0", "t1", "t2", "t3", "t4", "t5", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory" // 破坏描述符
    );

    factor_pipe0 = (127.f) / (float) sum_exp_sum;
    i_pipe1 = i_pipe0;
    factor_pipe1 = factor_pipe0;

    // ----- loop mainbody -----

    for (i_pipe0 = 1; i_pipe0 < dim_i; i_pipe0++) {
        // ----- pipeline 0 -----
        input_slice_pipe0 = &input[i_pipe0][0];

        __asm__ volatile (
            "mv a3, %[input_slice_pipe0]              \n" // a3 = input 起始地址
            "mv a5, %[j_itermax_m4]                \n"

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "li t1, -2147483648                 \n" // max_q[16] __attribute__((aligned(64))) = {-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648}
            "vmv.v.x v4, t1                     \n" // 将 v1 寄存器的所有元素初始化为 -2147483648                     



            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
            "vmax.vv v4, v4, v0       \n"   // v1 = max(v1, 0)

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            "vsetvli t0, zero, e32, m1  \n"
            "vmax.vv v4, v4, v5       \n"   // v1 = max(v1, 0)
            "vmax.vv v4, v4, v6       \n"   // v1 = max(v1, 0)
            "vmax.vv v4, v4, v7       \n"   // v1 = max(v1, 0)
            "vmv.v.x v0, t1             \n"
            "vredmax.vs v0, v4, v0      \n"
            "vmv.x.s %[max_q_max], v0          \n"

            : [max_q_max] "=r" (max_q_max)
            : [input_slice_pipe0] "r" (input_slice_pipe0), [j_itermax_m4] "r" (j_itermax_m4)  // 输入约束
            : "t0", "t1", "a3", "a4", "a5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory" // 破坏描述符
        );

        __asm__ volatile (
            "mv a3, %[input_slice_pipe0]              \n" // a3 = input 起始地址
            "mv a5, %[j_itermax_m4]                \n"
            "mv a6, %[max_q_max]                \n"
            "mv t2, %[qln2_inv]                 \n"
            "mv t3, %[qln2]                     \n"
            "mv t4, %[qb]                       \n"
            "mv t5, %[qc]                       \n"

            "li t1, 16                          \n" //

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "li t0, 0                           \n" //
            "vmv.v.x v4, t0           \n" // 加载 sum_exp 的 8 个元素到 v1

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
            "vsub.vx v0, v0, a6         \n" // q = input[i][j*8+k] - max_q_max;
            "vrsub.vi v8, v0, 0         \n" // -q
            "vmul.vx v8, v8, t2         \n" // -q * qln2_inv
            "vsra.vx v8, v8, t1         \n" // z = (acc_t) (-q * qln2_inv) >> 16;
            "vmul.vx v12, v8, t3         \n" // z * qln2
            "vadd.vv v0, v0, v12         \n" // qp = q + z * qln2;
            "vadd.vx v0, v0, t4         \n" // qp + qb
            "vmul.vv v0, v0, v0         \n" // (qp + qb)*(qp + qb)
            "vadd.vx v0, v0, t5         \n" // q_exp = (qp + qb)*(qp + qb) + qc;
            "vsra.vv v0, v0, v8         \n" // input[i][j*8+k] = q_exp >> z;
            "vadd.vv v4, v4, v0         \n" // sum_exp[k] += input[i][j*8+k];
            "vse32.v v0, (a3)           \n" // input[i][j*8+k] = q_exp >> z;

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            "vsetvli t0, zero, e32, m1  \n"
            "vadd.vv v4, v4, v5         \n"
            "vadd.vv v4, v4, v6         \n"
            "vadd.vv v4, v4, v7         \n"
            "vmv.v.x    v0, x0          \n"
            "vredsum.vs v0, v4, v0      \n"
            "vmv.x.s    %[sum_exp_sum], v0          \n"

            : [sum_exp_sum] "=r" (sum_exp_sum)
            : [input_slice_pipe0] "r" (input_slice_pipe0), [j_itermax_m4] "r" (j_itermax_m4), [max_q_max] "r" (max_q_max), [qln2_inv] "r" (qln2_inv), [qln2] "r" (qln2), [qb] "r" (qb), [qc] "r" (qc)  // 输入约束
            : "t0", "t1", "t2", "t3", "t4", "t5", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory" // 破坏描述符
        );

        factor_pipe0 = (127.f) / (float) sum_exp_sum;

        // ----- pipeline 1 -----

        input_slice_pipe1 = &input[i_pipe1][0];
        elem_t *output_slice = output + i_pipe1 * stride_c; 

        __asm__ volatile (
            "mv a3, %[input_slice_pipe1]              \n" // a3 = input 起始地址
            "mv a4, %[output_slice]                    \n" // a4 = max_q 起始地址
            "mv a5, %[j_itermax_store]                \n"
            "fmv.s f1, %[factor_pipe1]                 \n"

            "mv t2, %[elem_t_max]                 \n"
            "mv t3, %[elem_t_min]                     \n"

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "vle32.v v0, (a3)           \n" // x = input[i][j*8+k];
            "vfcvt.f.x.v v0, v0         \n" // int to float
            "vfmul.vf v0, v0, f1         \n" // x = x * factor_8[k];
            "vfcvt.x.f.v v0, v0         \n" // float to int
            "vmin.vx v0, v0, t2         \n" // x = x > elem_t_max ? elem_t_max :
            "vmax.vx v0, v0, t3         \n" // x < elem_t_min ? elem_t_min : x

            "vsetvli t0, zero, e16, m1  \n" 
            "vnclip.wi v6, v0, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
            "vnclip.wi v7, v2, 0          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))
            "vsetvli t0, zero, e8, m1  \n"
            "vnclip.wi v8, v6, 0          \n"
            
            "vse8.v v8, (a4)           \n" // input[i][j*8+k] = q_exp >> z;

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a4, a4, 64            \n"  // output的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            : // 输出寄存器（空）
            : [input_slice_pipe1] "r" (input_slice_pipe1), [output_slice] "r" (output_slice), [j_itermax_store] "r" (j_itermax_store), [factor_pipe1] "f" (factor_pipe1), [elem_t_max] "r" (elem_t_max), [elem_t_min] "r" (elem_t_min)  // 输入约束
            : "f1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "memory" // 破坏描述符
        );
        i_pipe1 = i_pipe0;
        factor_pipe1 = factor_pipe0;
    }

    // ----- finish -----

    input_slice_pipe1 = &input[i_pipe1][0];
    elem_t *output_slice = output + i_pipe1 * stride_c; 

    __asm__ volatile (
        "mv a3, %[input_slice_pipe1]              \n" // a3 = input 起始地址
        "mv a4, %[output_slice]                    \n" // a4 = max_q 起始地址
        "mv a5, %[j_itermax_store]                \n"
        "fmv.s f1, %[factor_pipe1]                 \n"

        "mv t2, %[elem_t_max]                 \n"
        "mv t3, %[elem_t_min]                     \n"

        "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

        "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
        "vle32.v v0, (a3)           \n" // x = input[i][j*8+k];
        "vfcvt.f.x.v v0, v0         \n" // int to float
        "vfmul.vf v0, v0, f1         \n" // x = x * factor_8[k];
        "vfcvt.x.f.v v0, v0         \n" // float to int
        "vmin.vx v0, v0, t2         \n" // x = x > elem_t_max ? elem_t_max :
        "vmax.vx v0, v0, t3         \n" // x < elem_t_min ? elem_t_min : x

        "vsetvli t0, zero, e16, m1  \n" 
        "vnclip.wi v6, v0, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnclip.wi v7, v2, 0          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vsetvli t0, zero, e8, m1  \n"
        "vnclip.wi v8, v6, 0          \n"
        
        "vse8.v v8, (a4)           \n" // input[i][j*8+k] = q_exp >> z;

        "addi a3, a3, 256            \n"  // input的下一个起始地址
        "addi a4, a4, 64            \n"  // output的下一个起始地址
        "addi a5, a5, -1            \n" // 列计数器 a5--
        "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

        : // 输出寄存器（空）
        : [input_slice_pipe1] "r" (input_slice_pipe1), [output_slice] "r" (output_slice), [j_itermax_store] "r" (j_itermax_store), [factor_pipe1] "f" (factor_pipe1), [elem_t_max] "r" (elem_t_max), [elem_t_min] "r" (elem_t_min)  // 输入约束
        : "f1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "memory" // 破坏描述符
    );

}

void layernorm_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    // printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);
    int j_itermax = dim_j / 16;
    int j_itermax_m4 = dim_j / 64;

    //vector layer norm
    for (size_t i = 0; i < dim_i; i++) {
        
        // for (size_t j = 0; j < dim_j/8; j++)
        // {
        //     for (size_t k = 0; k < 8; k++)
        //     {
        //         sum[k] += input[i][j*8+k];
        //     }
        // }
        
        acc_t *input_slice = &input[i][0];
        acc_t sum_sum = 0;
        __asm__ volatile (
            "mv a3, %[input_slice]              \n" // a3 = input 起始地址
            "mv a5, %[j_itermax_m4]                \n"

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "li t1, 0                   \n"
            "vmv.v.x v4, t1             \n" // 加载 sum 的 8 个元素到 v1

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
            "vadd.vv v4, v4, v0       \n"   // 

            "addi a3, a3, 256           \n"  // input的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "vadd.vv v4, v4, v5       \n"  
            "vadd.vv v4, v4, v6       \n"  
            "vadd.vv v4, v4, v7       \n"  
            "vmv.v.x v0, t1             \n"
            "vredsum.vs v0, v4, v0 \n" // 将 v4 中的所有元素相加，结果存储在 v0 中
            "vmv.x.s %[sum_sum], v0           \n"

            : [sum_sum] "=r" (sum_sum)
            : [input_slice] "r" (input_slice), [j_itermax_m4] "r" (j_itermax_m4)  // 输入约束
            : "t0", "t1", "a3", "a4", "a5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory" // 破坏描述符
        );


        acc_t t_mean = sum_sum / (acc_t)dim_j;
        acc_t total_err_sq_sum = 0;
        // for (size_t j = 0; j < dim_j/8; j++)
        // {
        //     for (size_t k = 0; k < 8; k++)
        //     {
        //         total_err_sq[k] += (input[i][j*8+k] - mean[k])*(input[i][j*8+k] - mean[k]);
        //     }
        // }
        __asm__ volatile (
            "mv a3, %[input_slice]              \n" // a3 = input 起始地址
            "mv a4, %[total_err_sq_sum]                    \n" // a4 = max_q 起始地址
            "mv a5, %[j_itermax_m4]                \n"
            "mv a6, %[t_mean]                \n"


            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "li t1, 0                           \n" //
            "vmv.v.x v4, t1                     \n"

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vle32.v v0, (a3)           \n" // 加载 input 的 8 个元素到 v0
            "vsub.vx v0, v0, a6         \n" // input[i][j*8+k] - mean[k]
            "vmul.vv v0, v0, v0         \n" // (input[i][j*8+k] - mean[k])*(input[i][j*8+k] - mean[k])
            "vadd.vv v4, v4, v0         \n" // total_err_sq[k] += (input[i][j*8+k] - mean[k])*(input[i][j*8+k] - mean[k]);

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            "vsetvli t0, zero, e32, m1  \n"
            "vadd.vv v4, v4, v5       \n"   // v1 = max(v1, 0)
            "vadd.vv v4, v4, v6       \n"   // v1 = max(v1, 0)
            "vadd.vv v4, v4, v7       \n"   // v1 = max(v1, 0)
            "vmv.v.x v0, t1             \n"
            "vredsum.vs v0, v4, v0      \n"
            "vmv.x.s %[total_err_sq_sum], v0           \n"
            : [total_err_sq_sum] "=r" (total_err_sq_sum)
            : [input_slice] "r" (input_slice), [j_itermax_m4] "r" (j_itermax_m4), [t_mean] "r" (t_mean)  // 输入约束
            : "t0", "t1", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory" // 破坏描述符
        );

        acc_t variance = total_err_sq_sum / (acc_t)dim_j;

        acc_t stddev_t = int_sqrt(variance);
        if (stddev_t == 0) stddev_t = 1;

        acc_scale_t stddev_f = 1 / ((acc_scale_t) stddev_t * scale);

        // for (size_t j = 0; j < dim_j/8; j++)
        // {
        //     for (size_t k = 0; k < 8; k++)
        //     {
        //         elem_t* c = output + i * stride_c + j*8+k;
        //         acc_t x = input[i][j*8+k];
        //         x -= mean[k];
        //         x /= stddev[k]*scale;
        //         // Clip result
        //         x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
        //         *c = x;
        //     }
        // }
        elem_t *output_slice = output + i * stride_c; 
        int j_itermax_store = dim_j / 64;
        __asm__ volatile (
            "mv a3, %[input_slice]              \n" // a3 = input 起始地址
            "mv a4, %[output_slice]                    \n" // a4 = max_q 起始地址
            "mv a5, %[j_itermax_store]                \n"
            "mv a6, %[res_input]                      \n"
            "fmv.s f1, %[stddev_f]                \n"

            "li t0, 0                           \n" //
            "mv t2, %[elem_t_max]                 \n"
            "mv t3, %[elem_t_min]                     \n"
            "mv t1, %[t_mean]                   \n"

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "vle32.v v0, (a3)           \n" // x = input[i][j*8+k];
            "vsub.vx v0, v0, t1         \n" //  x -= mean[k];

            // TODO x /= stddev[k]*scale;
            "vfcvt.f.x.v v0, v0         \n" // int to float
            "vfmul.vf v0, v0, f1        \n" // x /= stddev[k]*scale;
            "vfcvt.x.f.v v0, v0         \n" // float to int
            
            "vmin.vx v0, v0, t2         \n" // x = x > elem_t_max ? elem_t_max :
            "vmax.vx v0, v0, t3         \n" // x < elem_t_min ? elem_t_min : x

            "vsetvli t0, zero, e16, m1  \n" 
            "vnclip.wi v6, v0, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
            "vnclip.wi v7, v2, 0          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))
            "vsetvli t0, zero, e8, m1  \n"
            "vle8.v v9, (a6)            \n"
            "vnclip.wi v8, v6, 0          \n"
            "vsetvli t0, zero, e8, m1  \n"
            "vwadd.vv v10, v9, v8       \n"
            "vsetvli t0, zero, e16, m2  \n"
            "vmin.vx v10, v10, t2         \n" // x = x > elem_t_max ? elem_t_max :
            "vmax.vx v10, v10, t3         \n" // x < elem_t_min ? elem_t_min : x
            "vsetvli t0, zero, e8, m1  \n"
            "vnclip.wi v8, v10, 0          \n"
            
            "vse8.v v8, (a4)           \n" // input[i][j*8+k] = q_exp >> z;

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a6, a6, 64            \n"
            "addi a4, a4, 64            \n"  // output的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            : // 输出寄存器（空）
            : [input_slice] "r" (input_slice), [output_slice] "r" (output_slice), [j_itermax_store] "r" (j_itermax_store), [res_input] "r" (res_input), [stddev_f] "f" (stddev_f), [elem_t_max] "r" (elem_t_max), [elem_t_min] "r" (elem_t_min), [t_mean] "r" (t_mean)  // 输入约束
            : "f1", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory" // 破坏描述符
        );
        res_input += dim_j;
    }
}

void Gelu_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    // printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);

    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qb_inv = 1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    //Vector I-Gelu
    for (size_t i = 0; i < dim_i; i++) {

        // for (size_t j = 0; j < dim_j; j+=8) {
        //     for (size_t k = 0; k < 8; k++) {
        //         acc_t q = input[i][j+k];
        //         acc_t q_sign = q < 0 ? -1 : 1;
        //         acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
        //         acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
        //         acc_t q_erf = q_sign * q_poly;
        //         q = q * (q_erf + q1);
        //         elem_t* c = output + i * stride_c + j+k;
        //         // Clip result
        //         q = q > elem_t_max ? elem_t_max : (q < elem_t_min ? elem_t_min : q);
        //         *c = q;
        //     }
        // }

        acc_t *input_slice = &input[i][0];
        elem_t *output_slice = output + i * stride_c; 
        int j_itermax_store = dim_j / 64;
        __asm__ volatile (
            "mv a3, %[input_slice]              \n" // a3 = input 起始地址
            "mv a4, %[output_slice]                    \n" // a4 = max_q 起始地址
            "mv a5, %[j_itermax_store]                \n"

            "li t0, 0                           \n" //
            "mv t2, %[elem_t_max]                 \n"
            "mv t3, %[elem_t_min]                     \n"
            "mv t4, %[q1]                     \n"
            "mv t5, %[qb]                     \n"
            "mv t6, %[qb_inv]                     \n" // -qb
            "mv t1, %[qc]                     \n"

            "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素

            "1:                         \n" // for (size_t j = 0; j < dim_j/8; j++) 

            "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素
            "vle32.v v8, (a3)           \n" // x = input[i][j*8+k];
            "vmv.v.i  v4, 1           \n" 
            "vmslt.vi v0, v8, 0      \n" 
            "vmerge.vim v4, v4, -1, v0      \n" 
            "vmul.vv v0, v8, v4        \n" // abs(q)
            "vmin.vx v0, v0, t6       \n" // q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
            "vadd.vx v0, v0, t5       \n" // q_clipped + qb
            "vmul.vv v0, v0, v0      \n" // (q_clipped + qb)*(q_clipped + qb)
            "vadd.vx v0, v0, t1       \n" // q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
            "vmul.vv v4, v4, v0        \n" // q_erf = q_sign * q_poly;
            "vadd.vx v4, v4, t4         \n" // q_erf + q1
            "vmul.vv v8, v8, v4         \n" // q = q * (q_erf + q1);

            "vmin.vx v8, v8, t2         \n" // x = x > elem_t_max ? elem_t_max :
            "vmax.vx v8, v8, t3         \n" // x < elem_t_min ? elem_t_min : x

            "vsetvli t0, zero, e16, m2  \n" 
            "vnclip.wi v4, v8, 0          \n" // 
            "vsetvli t0, zero, e8, m1  \n"
            "vnclip.wi v8, v4, 0          \n"
            
            "vse8.v v8, (a4)           \n" // input[i][j*8+k] = q_exp >> z;

            "addi a3, a3, 256            \n"  // input的下一个起始地址
            "addi a4, a4, 64            \n"  // output的下一个起始地址
            "addi a5, a5, -1            \n" // 列计数器 a5--
            "bnez a5, 1b                \n" // 如果 a5 != 0，跳转到 row_loop

            : // 输出寄存器（空）
            : [input_slice] "r" (input_slice), [output_slice] "r" (output_slice), [j_itermax_store] "r" (j_itermax_store), [elem_t_max] "r" (elem_t_max), [elem_t_min] "r" (elem_t_min), [q1] "r" (q1) , [qb] "r" (qb), [qb_inv] "r" (qb_inv), [qc] "r" (qc)
            : "s2", "s3", "s4", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a3", "a4", "a5", "a6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "memory" // 破坏描述符
        );
        // exit(0);
    }
}

void scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I)
{
    //shift 8 是可以有特殊优化的，只load 7bit+1bit，就能完成relu和shift
    __asm__ volatile (
        // 初始化寄存器
        "li s6, 1                   \n" // 设置vxrm为rnd_to_nearest_even = 1
        "li s7, 2                   \n"
        "csrrw zero, vxrm, s6       \n" // 设置vxrm为rnd_to_nearest_even
        "mv a3, %[input]            \n" // a3 = input 起始地址
        "mv a4, %[output]           \n" // a4 = output 起始地址
        "mv a5, %[stride_input]     \n" // a5 = input 的行步长
        "mv a6, %[stride_output]    \n" // a6 = output 的行步长
        "mv a7, %[shift_scale]      \n" // a7 = shift_scale
        
        "mv t1, %[dim_I]            \n" // 行计数器 t1 = 64
        // "li s6, 1                   \n" // 设置vxrm为rnd_to_nearest_even = 1
        "li t2, 127                 \n" // relu_max = 127
        "li s5, -128                \n" // relu_min = -128
        "li s9, 4                   \n" // 循环展开四次
        "li t3, 0                   \n" // input列寄存器 a3 + 32
        "li t4, 0                   \n" // input列寄存器 a3 + 64
        "li t5, 0                   \n" // input列寄存器 a3 + 96
        "li t6, 0                   \n" // input列寄存器 a3 + 128
        "li s2, 0                   \n" // input列寄存器 a3 + 160
        "li s3, 0                   \n" // input列寄存器 a3 + 192
        "li s4, 0                   \n" // input列寄存器 a3 + 224
        "li t0, 0                   \n" // output列寄存器 a4 + 32

        // 加载数据，避免load指令重复依赖


        // 设置向量长度，假设 vlen 为 256 位（32 个 8-bit 元素）

        "1:                      \n" // 外层循环标签 row_loop
        "addi t6, a3, 0          \n" // （下0行的起始位置）
        "add s8, a5, a3         \n" //

        "vsetvli t0, zero, e32, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vle32.v  v0, (t6)            \n" // 加载 input 的前 8 个元素到 v0
        "vle32.v  v4, (s8)            \n" // 加载 input 的下 8 个元素到 v4

        "add t6, s8, a5          \n" //
        "add s8, t6, a5          \n" //


        "vle32.v v8, (t6)            \n" // 加载 input 的前 8 个元素到 v0
        "vle32.v v12, (s8)            \n" // 加载 input 的下 8 个元素到 v4

        //-----//-load end！！-----------------------------//

        // 向量操作，ReLU 和移位
        // "csrrw zero, vxrm, s6       \n" // 设置vxrm为rnd_to_nearest_even
        "vsetvli t0, zero, e16, m8  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vnclip.wx v0,   v0, a7          \n"  // v8 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vmin.vx v0,  v0, t2          \n"  // v8 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vmax.vx v0,  v0, s5          \n"  // v8 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vsetvli t0, zero, e8, m4  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次，缩小到8bit
        
        "vnclip.wi v0,   v0, 0          \n"  // v8 = cat (clip(v1 >> 8), clip(v2 >> 8))

        // 将数据转换为 8-bit，并存储到v9-v10
        "add t3, a4, a6        \n" // 
        "add t5, t3, a6        \n" // 
        "add s3, t5, a6        \n" // 

        // 存储数据（连续写回）
        "vsetvli t0, zero, e8, m1  \n"
        "vse8.v v0, (a4)       \n" // 
        "vse8.v v1, (t3)       \n" // 
        "vse8.v v2, (t5)       \n" // 
        "vse8.v v3, (s3)       \n" //
        // store更新行指针
        "add a4, s3, a6             \n" // a4 前进 stride_c 字节（下一行的起始位置
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）

        "addi t1, t1, -4            \n" // 行计数器 t1--
        "bge t1, s9, 1b             \n" // 如果 t1 != 0，跳转到 row_loop

        "beqz t1, 3f                \n" // 如果 t1 == 0，跳转到 end_loop

        "2:                         \n" // dim_I不是4的倍数时处理边缘循环
        // 加载数据，避免load指令重复依赖
        "addi t3, a3, 64             \n" // t3 = a3 + 64
        "addi t4, a3, 128             \n" // t4 = a3 + 128
        "addi t5, a3, 192             \n" // t5 = a3 + 192

        "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vle32.v v0, (a3)            \n" // 加载 input 的前 8 个元素到 v0
        "vle32.v v1, (t3)            \n" // 加载 input 的下 8 个元素到 v1
        "vle32.v v2, (t4)            \n" // 加载 input 的下 8 个元素到 v2
        "vle32.v v3, (t5)            \n" // 加载 input 的下 8 个元素到 v3

        // 向量操作，ReLU 和移位
        "vsetvli t0, zero, e16, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vnclip.wx v8, v0,  a7          \n"  // v8  = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnclip.wx v9, v2, a7          \n" // v9 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vmin.vx v8, v8, t2       \n"   // v1 = minu(v1, 127) 
        "vmin.vx v9, v9, t2       \n" // v2 = minu(v2, 127)

        "vmax.vx v8, v8, s5       \n"   // v1 = max(v1, 0)
        "vmax.vx v9, v9, s5       \n" // v2 = max(v2, 0)

        "vsetvli t0, zero, e8, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次，缩小到8bit
        "vnclip.wi v12, v8, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))

        // 将数据转换为 8-bit，并存储到v9-v10

        // 存储数据（连续写回）
        "vse8.v v12, (a4)       \n" // 存储 v20 到 output（32 个元素）

        // 更新行指针
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a4, a4, a6             \n" // a4 前进 stride_c 字节（下一行的起始位置）
        "addi t1, t1, -1            \n" // 行计数器 t1--
        "bnez t1, 2b                \n" // 如果 t1 != 0，跳转到 row_loop

        "3:                         \n" // end_loop


        : // 输出寄存器（空）
        : [input] "r" (input), [output] "r" (output), [stride_input] "r" (stride_input), [stride_output] "r" (stride_output), [shift_scale] "r" (shift_scale), [dim_I] "r" (dim_I)  // 输入约束
        : "t0", "t1","t2","t3","t4","t5","t6","s2","s3","s4","s5","s6","s7","s8","s9", "a3", "a4", "a5","a6","v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory" // 破坏描述符
    );
}

void scale_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c) {
    scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x(input, output, 3072 * 4, stride_c,0, dim_j);
}

void MATMUL_MARCO_ISSUE()
{
    // printf("MATMUL_MARCO_ISSUE\n");
    // exit(1);
}

void CUTE_TASK_END(uint64_t task_id)
{
    // printf("waiting for task end\n");
    //等待任务结束
    uint64_t finish_tag = 1 << task_id;
    uint64_t res1 = cute_marco_inst_fifo_finish_search();
    while(!(res1&finish_tag))
    {
        // printf("Waiting for finish task_id = %d\n",task_id);
        res1 = cute_marco_inst_fifo_finish_search();
    }
    cute_marco_inst_fifo_dequeue();
    // printf("Task End\n");
    return;
}

int MATMUL_MARCO_SEARCH()
{
    // printf("MATMUL_MARCO_SEARCH\n");
    return (rand()%100)<17;
}

static void matmul_cute(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,int transpose_result) {

// scale = 1.0;
  const int no_bias = D == NULL;
  //输出所有输入参数
    // printf("transA:%d,transB:%d,DIM_I:%d,DIM_J:%d,DIM_K:%d,stride_A:%d,stride_B:%d,stride_D:%d,stride_C:%d\nA_scale_factor:%f,B_scale_factor:%f,D_scale_factor:%d,act:%d,scale:%f,bert_scale:%f,repeating_bias:%d\n",transA,transB,DIM_I,DIM_J,DIM_K,stride_A,stride_B,stride_D,stride_C,A_scale_factor,B_scale_factor,D_scale_factor,act,scale,bert_scale,repeating_bias);
  //如果不是layernorm或者softmax切成64,64,K的小块，然后每次完成计算，调用向量算子。
  //如果是layernorm或者softmax，或者不是64,M,K的小块，然后调用向量算子。

  if(!(DIM_I % 64 == 0 && DIM_J % 64 == 0 && DIM_K % 64 == 0))
  {
    printf("Can't Till Now!");
    exit(1);
  }

  if(DIM_J > 3072 && (act == LAYERNORM || act == SOFTMAX))
  {
    printf("DIM_J too large!");
    exit(1);
  }

  if((act == LAYERNORM || act == SOFTMAX) && transpose_result)
  {
    printf("LAYERNORM and SOFTMAX dont support transport!");
    exit(1);
  }

  void (*afater_operation)(acc_t *,int,int,elem_t *,acc_scale_t,int) = NULL;

  switch (act) {
    case NO_ACTIVATION:
      afater_operation = scale_after_operation;
      break;
    case RELU:
      afater_operation = NULL;
      break;
    case LAYERNORM:
      afater_operation = layernorm_after_operation;
      break;
    case IGELU:
      afater_operation = Gelu_after_operation;
      break;
    case SOFTMAX:
      afater_operation = softmax_after_operation;
      break;
    default:
      afater_operation = scale_after_operation;
      break;
  }
  
  // printf("!!\n[matmul_cute] START!!\n!!\n");
  if(act != LAYERNORM && act != SOFTMAX)
  {
    int Tile_I = DIM_I / 64;
    int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = 64;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = 3072 * 4;
    int Application_stride_D = stride_D;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = repeating_bias;
    int Is_Zero_Load = no_bias;
    uint64_t bias_type = Is_Zero_Load ? TaskTypeTensorZeroLoad : (Is_repeating_row ? TaskTypeTensorRepeatRowLoad : TaskTypeTensorLoad);


    uint64_t wait_after_operation_cute_task_id = 0;
    uint64_t wait_after_operation_cute_task_id_pre = 0;

    elem_t* Tile_A = A;
    elem_t* Tile_B = B;
    acc_t * Tile_C = CUTE_result[CUTE_result_index];
    acc_t * Tile_D = D;


    //后操作的函数指针，返回值是void
    
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */
    // MATMUL_MARCO_ISSUE();
    wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C,
                Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);
    CUTE_TASK_END(wait_after_operation_cute_task_id_pre);

    int i = 0;
    int j = 1;
    int pre_i = 0;
    int pre_j = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
    for (i=0;i<Tile_I;i++)
    for (j=(i==0?1:0);j<Tile_J;j++)
    {
        //等待CUTE任务完成
        // while(acc_not_finish)
        // {
        //     /*
        //     cute 完成查询
        //     */
        //    //假查询
        // }
        MATMUL_MARCO_SEARCH();
        // exit(0);

        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*Tile_J+j,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A;
        Tile_B = B + j * 64 * stride_B;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = no_bias ? D : (repeating_bias ? D + j * 64 : D + i * 64 * stride_D + j * 64);
        /*
        cute 配置
        cute 指令发射
        */
        MATMUL_MARCO_ISSUE();
        wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C,
                Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);
        
        //执行当前任务的CPU的向量后操作任务
        CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
        afater_operation(CUTE_result[CUTE_result_index],64,64,(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64),bert_scale,stride_C);
        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*DIM_J+j,DIM_K);
        // printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
        pre_j = j;
    }

    afater_operation(CUTE_result[CUTE_result_index],64,64,(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64),bert_scale,stride_C);
    // printf("[Final][Vec]Vector Operation %s Finish\n",activation_name(act));
    

  }else
  {
    int Tile_I = DIM_I / 64;
    // int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = DIM_J;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = 3072 * 4;
    int Application_stride_D = stride_D;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = repeating_bias;
    int Is_Zero_Load = no_bias;

    uint64_t bias_type = Is_Zero_Load ? TaskTypeTensorZeroLoad : (Is_repeating_row ? TaskTypeTensorRepeatRowLoad : TaskTypeTensorLoad);


    uint64_t wait_after_operation_cute_task_id_pre = 0;

    elem_t* Tile_A = A;
    elem_t* Tile_B = B;
    acc_t * Tile_C = CUTE_result[CUTE_result_index];
    acc_t * Tile_D = D;


    //后操作的函数指针，返回值是void
    
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */
   MATMUL_MARCO_ISSUE();
   wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C,
                Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);
  CUTE_TASK_END(wait_after_operation_cute_task_id_pre); 
    int i = 1;
    int pre_i = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
    for (i=1;i<Tile_I;i++)
    {
        //等待CUTE任务完成
        // while(acc_not_finish)
        // {
        //     /*
        //     cute 完成查询
        //     */
        //    //假查询
        // }
        MATMUL_MARCO_SEARCH();

        // CUTE_TASK_END(wait_after_operation_cute_task_id_pre);

        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*%d*%d\n",i,DIM_J,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A;
        Tile_B = B;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = no_bias ? D : (repeating_bias ? D : D + i * 64 * stride_D);
        /*
        cute 配置
        cute 指令发射
        */
       MATMUL_MARCO_ISSUE();
       wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C,
                Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);

        
        //执行当前任务的CPU的向量后操作任务
                CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
        afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),bert_scale,stride_C);
        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*DIM_J+j,DIM_K);
        // printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }

    afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),bert_scale,stride_C);
    // printf("[Final][Vec]Vector Operation %s Finish\n",activation_name(act));
    
  }
}


static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type,int cute_transpose_B) {

    matmul_cute(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias,cute_transpose_B);
}

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf)
{
    const int hidden_dim_compressed = hidden_dim / compression_factor;
    const int hidden_dim_per_head = hidden_dim_compressed / num_heads;


    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        // printf("[Get Attention's Out!]\n");
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * seq_len * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/seq_len, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            CPU,0);
    }


}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf)
{
    // out = FF1(input)
    // out = GELU(out)
    // printf("[FF1!With Igelu]\n");
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU,0);

    gemmini_fence();

    // out = FF2(out)
    // out = LN(out)

    // printf("[FF2!With LayerNorm]\n");
    res_input = input;
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out,
        /*stride_A=*/expansion_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 1,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU,0);


    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,
        Q_buf, K_buf, V_buf,
        attn_buf, out_buf);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf);
    }


    uint64_t end = read_cycles();

    return end - start;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    \
    static elem_t Wqkvo[4][hidden_dim][hidden_dim] __attribute__((aligned(256))); \
    static elem_t Wqkvo_cross[4][hidden_dim][hidden_dim] __attribute__((aligned(256))); \
    static elem_t ff_w[2][hidden_dim*expansion_dim] __attribute__((aligned(256))); \
    static acc_t ff1_b[expansion_dim] __attribute__((aligned(256))); \
    static acc_t ff2_b[hidden_dim] __attribute__((aligned(256))); \
    static elem_t QKV_buf[3][seq_len][hidden_dim] __attribute__((aligned(256)));\
    static elem_t attn_buf[num_heads][seq_len][seq_len] __attribute__((aligned(256)));\
    static elem_t out_buf[seq_len][expansion_dim] __attribute__((aligned(256)));\
    static elem_t resadd1_buf[seq_len][hidden_dim] __attribute__((aligned(256)));\
    static elem_t resadd2_buf[seq_len][hidden_dim] __attribute__((aligned(256)));\
    \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            compression_factor, \
            \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define PRINT_ENCODER_DECODER(name, is_encoder, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor) { \
    static elem_t input[seq_len][hidden_dim] __attribute__((aligned(256))); \
    static elem_t enc_out[seq_len][hidden_dim] __attribute__((aligned(256))); \
    static elem_t output[seq_len][hidden_dim] __attribute__((aligned(256))); \
    \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
    \
    uint64_t cycles = ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, is_encoder ? NULL : enc_out, output); \
    \
    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n", \
            name, type_str, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor); \
    printf("%s cycles: %lu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {

    // uint64_t start = read_cycles();
    // gemmini_flush(0);

    // PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
    //         /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128, /*compression_factor=*/1);

    // uint64_t end = read_cycles();
  // printf("matmul cycles %d\n", end - start);

        PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
                /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/128, /*compression_factor=*/1);

    exit(0);
}

