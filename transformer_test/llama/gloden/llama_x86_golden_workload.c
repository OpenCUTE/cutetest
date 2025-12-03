#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "data.h"
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

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



void print_array_2d_elem(float *array, int len1, int len2) {
    printf("len1: %d, len2: %d\n", len1, len2);
    printf("{\n");
    for (int i = 0; i < len1; i++) {
        printf("{");
        for (int j = 0; j < len2; j++) {
            printf("%f,", array[i * len2 + j]);
        }
        printf("},\n");
    }
    printf("};\n");
}

void print_array_3d_elem(float *array, int len1, int len2, int len3) {
    printf("{\n");
    for (int i = 0; i < len1; i++) {
        printf("{\n");
        for (int j = 0; j < len2; j++) {
            printf("{");
            for (int k = 0; k < len3; k++) {
                // fprintf(fp, "%d,", array[i][j][k]);
                printf("%f,", array[i * len2 * len3 + j * len3 + k]);
            }
            printf("},\n");
        }
        printf("},\n");
    }
    printf("};\n");
}


static void resadd_cpu(const size_t I, const size_t J,
        float * A,
        float * B,
        float * C) {


    int size = I * J;
    for (size_t i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// acc_t __attribute__((aligned(512))) CUTE_result[2][64][3072];//double buffer
// int CUTE_result_index = 0;

void RMSnorm(float * input, float * weight, int dim_i, int dim_j, float * output, int stride_c)
{
    //输出所有函数输入
    //
    // printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm vector work,for layer norm. layer length: %d, n layers: %d\n",dim_i, dim_j,dim_i, dim_j);
    // printf("dim_i:%d,dim_j:%d,stride_c:%d\n",dim_i,dim_j,stride_c);

    for (int i = 0; i < dim_i; i++) {
        float sum = 0;
        for (int j = 0; j < dim_j; j++) {
            sum += input[i * dim_j + j] * input[i * dim_j + j];
        }
        float mean = RMS_EPSILON + sum / dim_j;
        float stddev = sqrt(mean);
        if (stddev == 0) stddev = 1;

        for (int j = 0; j < dim_j; j++) {
            output[i * dim_j + j] = input[i * dim_j + j] / stddev * weight[j];
        }
    }
}

// void einsum()
float idx_theta_buf[MAX_CTX_LEN][KEY_DIMENSION/2] __attribute__((aligned(256)));

void rope(float *input, int pos, int dim_i, int dim_j, float *output, int stride_c) {
    //per generation all block use the same theta
    //同一个generation内，使用同一个位置编码，且每次计算新的位置编码都可以转换成一次简单的乘加操作来完成
    for (int i = 0; i < dim_i; i++) {
        for (int j = 0; j < dim_j / 2; j++) {
            idx_theta_buf[i][j] = (i + pos) * rope_theta[j];
        }
    }

    //rope计算就是对所有的输入进行，逐元素的乘累加操作，sin(θ) = sin(α)cos(φ) + cos(α)sin(φ)，cos(θ)同理的计算可以提前算好,sin(α)和cos(α)则是上一轮的结果，执行两次乘累加即可计算出cos(φ)和sin(φ)则是固定值。
    //计算玩theta矩阵后，计算output就很简单了。就是两次乘累加操作。
    for (int i = 0; i < dim_i; i++) {
        for (int j = 0; j < dim_j/2; j++) {
            float theta = idx_theta_buf[i][j];
            float cos_theta = cos(theta);
            float sin_theta = sin(theta);
            output[i * stride_c + j]            = input[i * dim_j + j * 2] * cos_theta - input[i * dim_j + j * 2 + 1] * sin_theta;
            output[i * stride_c + j + dim_j/2]  = input[i * dim_j + j * 2] * sin_theta + input[i * dim_j + j * 2 + 1] * cos_theta;
        }
    }
}

void softmax(float *input, int dim_i, int dim_j, float *output, int stride_c) {
    // 输出所有函数输入
    // printf("dim_i:%d,dim_j:%d,stride_c:%d\n", dim_i, dim_j, stride_c);

    for (int i = 0; i < dim_i; i++) {
        float sum_exp = 0;
        for (int j = 0; j < dim_j; j++) {
            sum_exp += exp(input[i * dim_j + j]);
        }

        for (int j = 0; j < dim_j; j++) {
            output[i * stride_c + j] = exp(input[i * dim_j + j]) / sum_exp;
        }
    }
}

void silu(float *input, int dim_i, int dim_j, float *output, int stride_c) {

    for (int i = 0; i < dim_i; i++) {
        for (int j = 0; j < dim_j; j++) {
            float x = input[i * dim_j + j];
            output[i * stride_c + j] = x / (1 + exp(-x));
        }
    }
}


void MATMUL_MARCO_ISSUE()
{
    printf("MATMUL_MARCO_ISSUE\n");
    // exit(1);
}

int MATMUL_MARCO_SEARCH()
{
    printf("MATMUL_MARCO_SEARCH\n");
    return (rand()%100)<17;
}

static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        float *A, float *B, float *D, float *C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int act, bool repeating_bias, int transpose_result) {

    const int no_bias = D == NULL;

    for (size_t i = 0; i < DIM_I; i++) {
        for (size_t j = 0; j < DIM_J; j++) {

            size_t bias_row = repeating_bias ? 0 : i;
            float result = no_bias ? 0 : D[bias_row * stride_D + j];

            for (size_t k = 0; k < DIM_K; k++) {
                result += A[i * stride_A + k] * B[j * stride_B + k];
            }

            if (transpose_result) {
                C[j * stride_C + i] = result;
            } else {
                C[i * stride_C + j] = result;
            }
        }
    }
}

uint64_t llama_block(
        int input_size, int d, int dk, int dv, int head_q, int head_kv, int dffn,


        float * hidden_states, float * q_buf, float * k_buf, float * v_buf, float * o_buf, float * o_transpose, float * k_interleave, float * v_interleave, float * mat_buf, float * scores_buf,
        float * gate, float * up, float * down)
{

    // uint64_t start = read_cycles();
    // input_size=6, d=2048, dk=64, dv=64, head_q=32, head_kv=8
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm_input\n",input_size, d);
    RMSnorm(identity, attn_norm_weight, input_size, d, hidden_states, d * 4);

    // q: [nhq, sl, dk] = RoPE([sl, d] @ [nhq, dk, d])
    // q = self.rope(hidden_states @ self.attn_q_weight[i].transpose(1, 2), pos=self.ctx_len)
    for (int i = 0; i < head_q; i++) {
        float *A = hidden_states;
        float *B = attn_q_weight[i];
        float *C = q_buf + i * input_size * dk;
        float *sub_mat_buf = mat_buf + i * input_size * dk;
        printf("[WorkLoad-(%5d,%5d,%5d)Matmul]attention_Q\n",input_size, dk,d);
        matmul_cpu(input_size, dk, d,
            A, B, NULL, sub_mat_buf,
            d, d, 0, dk, 0, false, false);
        printf("[WorkLoad-(%5d,%5d,*****)elementWise]rope_Q\n",input_size, dk);
        rope(sub_mat_buf, 0, input_size, dk, C, dk);
    }

    // k: [nhkv, sl, dk] = RoPE([sl, d] @ [nhkv, dk, d])
    // k = self.rope(hidden_states @ self.attn_k_weight[i].transpose(1, 2), pos=self.ctx_len)
    for (int i = 0; i < head_kv; i++) {
        float *A = hidden_states;
        float *B = attn_k_weight[i];
        float *C = k_buf + i * input_size * dk;
        float *sub_mat_buf = mat_buf + i * input_size * dk;
        printf("[WorkLoad-(%5d,%5d,%5d)Matmul]attention_K\n",input_size, dk,d);
        matmul_cpu(input_size, dk, d,
            A, B, NULL, sub_mat_buf,
            d, d, 0, dk, 0, false, false);
        printf("[WorkLoad-(%5d,%5d,*****)elementWise]rope_K\n",input_size, dk);
        rope(sub_mat_buf, 0, input_size, dk, C, dk);
    }

    // v: [nhkv, sl, dv] = [sl, d] @ [nhkv, dv, d]
    // v = hidden_states @ self.attn_v_weight[i].transpose(1, 2)  
    for (int i = 0; i < head_kv; i++) {
        float *A = hidden_states;
        float *B = attn_v_weight[i];
        float *C = v_buf + i * input_size * dv;
        printf("[WorkLoad-(%5d,%5d,%5d)Matmul]attention_V\n",input_size, dk,d);
        matmul_cpu(input_size, dv, d,
            A, B, NULL, C,
            d, d, 0, dv, 0, false, false);
    }

    // printf("hidden_states @ self.attn_q_weight = ");
    // print_array_3d_elem(mat_buf, head_q, input_size, dk);

    // printf("q_buf = ");
    // print_array_3d_elem(q_buf, head_q, input_size, dk);

    // printf("k_buf = ");
    // print_array_3d_elem(k_buf, head_kv, input_size, dk);

    // printf("v_buf = ");
    // print_array_3d_elem(v_buf, head_kv, input_size, dv);

    // k = k.repeat_interleave(self.n_head_q // self.n_head_kv, dim=0)                     
    // v = v.repeat_interleave(self.n_head_q // self.n_head_kv, dim=0)
    //....这里完全不需要真的复制n份...重复张量
    int inter_num = head_q / head_kv;
    for (int i = 0; i < head_kv; i++) {
        float *k = k_buf + i * input_size * dk;
        float *v = v_buf + i * input_size * dv;
        for (int j = 0; j < inter_num; j++) {
            float *k_interleave_i = k_interleave + (i * inter_num + j) * input_size * dk;
            float *v_interleave_i = v_interleave + (i * inter_num + j) * input_size * dv;
            for (int k_idx = 0; k_idx < input_size; k_idx++) {
                for (int dk_idx = 0; dk_idx < dk; dk_idx++) {
                    k_interleave_i[k_idx * dk + dk_idx] = k[k_idx * dk + dk_idx];
                }
                for (int dv_idx = 0; dv_idx < dv; dv_idx++) {
                    v_interleave_i[dv_idx * input_size + k_idx] = v[k_idx * dv + dv_idx];
                }
            }
        }
    }

    // printf("k_interleave = ");
    // print_array_3d_elem(k_interleave, head_q, input_size, dk);

    // printf("v_interleave = ");
    // print_array_3d_elem(v_interleave, head_q, input_size, dv);

    // scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
    int factor = sqrt(dk);//这里估计对阶码直接进行减法就好了，因为这个dk大概率是个2的2n次方

    for (int i = 0; i < head_q; i++) {
        float *A = q_buf + i * input_size * dk;
        float *B = k_interleave + i * input_size * dk;
        float *C = scores_buf + i * input_size * input_size;
        printf("[WorkLoad-(%5d,%5d,%5d)Matmul]scores\n",input_size, input_size,dk);
        matmul_cpu(input_size, input_size, dk,
            A, B, NULL, C,
            dk, dk, 0, input_size, 0, false, false);
        for (int j = 0; j < input_size * input_size; j++) {
            C[j] /= factor;
        }
    }

    // scores = scores.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))
    // masked_fill?别算就好了，你mask干嘛
    for (int i = 0; i < head_q; i++) {
        float *A = scores_buf + i * input_size * input_size;
        for (int j = 0; j < input_size; j++) {
            for (int k = j + 1; k < input_size; k++) {
                A[j * input_size + k] = -INFINITY;
            }
        }
    }

    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]sofamax_scores\n",head_q * input_size,input_size);
    softmax(scores_buf, head_q * input_size, input_size, scores_buf, input_size);

    // printf("scores_buf = ");
    // print_array_3d_elem(scores_buf, head_q, input_size, input_size);

    // o = torch.bmm(scores, v)
    for (int i = 0; i < head_q; i++) {
        float *A = scores_buf + i * input_size * input_size;
        float *B = v_interleave + i * input_size * dv;
        float *C = o_buf + i * input_size * dv;
        printf("[WorkLoad-(%5d,%5d,%5d)Matmul]scores_Output\n",input_size, dv,input_size);
        matmul_cpu(input_size, dv, input_size,
            A, B, NULL, C,
            input_size, input_size, 0, dv, 0, false, false);
    }

    // printf("o_buf = ");
    // print_array_3d_elem(o_buf, head_q, input_size, dv);
    printf("[WorkLoad-(%5d,%5d,%5d)Transpose_Skip]output_Resheap\n",input_size, dv,input_size);
    for (int i = 0; i < head_q; i++) {
        for (int j = 0; j < input_size; j++) {
            float *o_transpose_i = o_transpose + j * head_q * dv + i * dv;
            float *o_buf_i = o_buf + i * input_size * dv + j * dv;
            for (int dv_idx = 0; dv_idx < dv; dv_idx++) {
                o_transpose_i[dv_idx] = o_buf_i[dv_idx];
            }
        }
    }

    // hidden_states = identity + o.transpose(0, 1).reshape(-1, self.n_head_q * self.d_v) @ self.attn_o_weight[i].T
    printf("[WorkLoad-(%5d,%5d,%5d)Matmul]attention_Output\n",input_size,d, head_q * dv);
    matmul_cpu(input_size, d, head_q * dv,
            o_transpose, attn_o_weight, NULL, hidden_states,
            head_q * dv, head_q * dv, 0, d, 0, false, false);

    printf("[WorkLoad-(%5d,%5d,*****)ResAdd]ResAdd_Output\n",input_size,d);
    resadd_cpu(input_size, d, identity, hidden_states, hidden_states);

    // printf("hidden_states = ");
    // print_array_2d_elem(hidden_states, input_size, d);


    // identity = hidden_states
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < d; j++) {
            identity[i][j] = hidden_states[i * d + j];
        }
    }

    // hidden_states = self.rmsnorm(hidden_states, self.ffn_norm_weight[i])
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm_Resout\n",input_size, d);
    RMSnorm(hidden_states, ffn_norm_weight, input_size, d, hidden_states, d * 4);
    // printf("hidden_states after ffn rmsnorm = ");
    // print_array_2d_elem(hidden_states, input_size, d);

    // gate = hidden_states @ self.ffn_gate_weight[i].T
    printf("[WorkLoad-(%5d,%5d,%5d)Matmul]FFN_Gate\n",input_size, dffn,d);
    matmul_cpu(input_size, dffn, d,
            hidden_states, ffn_gate_weight, NULL, gate,
            d, d, 0, dffn, 0, false, false);

    // gate = torch.nn.functional.silu(gate)
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]silu_FFNGate\n",input_size, dffn);
    silu(gate, input_size, dffn, gate, dffn);

    // printf("gate = ");
    // print_array_2d_elem(gate, input_size, dffn);

    // up = hidden_states @ self.ffn_up_weight[i].T
    printf("[WorkLoad-(%5d,%5d,%5d)Matmul]FFN_UP\n",input_size, dffn,d);
    matmul_cpu(input_size, dffn, d,
            hidden_states, ffn_up_weight, NULL, up,
            d, d, 0, dffn, 0, false, false);

    // printf("up = ");
    // print_array_2d_elem(up, input_size, dffn);

    int size = input_size * dffn;
    printf("[WorkLoad-(%5d,%5d,*****)ElementWise]Gate_UP_Multi\n",input_size, dffn);
    for (int i = 0; i < size; i++) {
        gate[i] *= up[i]; //这个放在权重里头不就好了？
    }

    // down = (gate * up) @ self.ffn_down_weight[i].T
    printf("[WorkLoad-(%5d,%5d,%5d)Matmul]FFN_DOWN\n",input_size, d,dffn);
    matmul_cpu(input_size, d, dffn,
            gate, ffn_down_weight, NULL, down,
            dffn, dffn, 0, d, 0, false, false);

    // printf("down = ");
    // print_array_2d_elem(down, input_size, d);

    // hidden_states = identity + down
    printf("[WorkLoad-(%5d,%5d,*****)ResAdd]ResAdd_FFN_OUT\n",input_size,d);
    resadd_cpu(input_size, d, identity, down, hidden_states);

    // uint64_t end = read_cycles();

    return 0;
}


#define LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn) ({ \
    static float hidden_states[input_size][d];\
    static float q_buf[head_q][input_size][dk]; \
    static float k_buf[head_kv][input_size][dk]; \
    static float v_buf[head_kv][input_size][dv]; \
    static float o_buf[head_q][input_size][dv]; \
    static float o_transpose[input_size][head_q][dv]; \
    static float k_interleave[head_q][input_size][dk]; \
    static float v_interleave[head_q][input_size * dv]; \
    static float mat_buf[head_q][input_size][dk]; \
    static float scores_buf[head_q][input_size][input_size]; \
    static float gate[input_size][dffn]; \
    static float up[input_size][dffn]; \
    static float down[input_size][d]; \
    uint64_t cycles = llama_block( \
            input_size, d, dk, dv, head_q, head_kv, dffn, \
            hidden_states, q_buf, k_buf, v_buf, o_buf, o_transpose, k_interleave, v_interleave, mat_buf, scores_buf, \
            gate, up, down \
    ); \
    \
    cycles; \
})

#define PRINT_LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn) { \
    \
    printf("input_size=%d, d=%d, dk=%d, dv=%d, head_q=%d, head_kv=%d\n", \
            input_size, d, dk, dv, head_q, head_kv); \
    uint64_t cycles = LLAMA(input_size, d, dk, dv, head_q, head_kv, dffn); \
}

int main (int argc, char * argv[]) {

    // gemmini_flush(0);

    PRINT_LLAMA(/*input_size=*/INPUT_SIZE, /*d=*/EMBEDING_DIMENSION, /*dk=*/KEY_DIMENSION, /*dv=*/VALUE_DIMENSION, /*head_q=*/N_HEAD_Q, /*head_kv=*/N_HEAD_KV, /*dffn=*/FFN_DIMENSION);

}

