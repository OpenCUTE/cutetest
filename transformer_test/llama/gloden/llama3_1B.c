#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
// #include "easy_test_data.h"
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "cuteMarcoinstHelper.h"

#define LAYEROPT 2048
#define FUSEOPT 1024
#define NO_ACTIVATION 0
#define DEQUANT 1
#define ROPE 2
#define CVRT_TO_BF16 3
#define SOFTMAX 4
#define RMSNORM 5
#define RESADD 6
#define SILU 7
#define HADAMARD_PRODUCT 8
#define PER_TOKEN_QUANT 9                   //for smoothquant do quant
#define KVSCALE 10
#define MASKED_SOFTMAX 11
#define QUANTSTAGE1 12                      //for smoothquant max abs
#define FUSE_DEQUANT_ROPE_BF16CVRT              (FUSEOPT + 1)   //dequant+rope+bf16cvrt for proj_q,proj_k to score
#define FUSE_DEQUANT_BF16CVRT                   (FUSEOPT + 2)   //dequant+bf16cvrt for proj_v to attention
#define FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT    (FUSEOPT + 3)   //softmax+bf16cvrt+KVSCALE for score to attention
#define FUSE_DEQUANT_RESADD_RMSNORM_QUANT       (FUSEOPT + 4)   //dequant+resadd+rmsnorm+quant for proj_o to ffn_gate,ffn_up
#define FUSE_DEQUANT_SILU                       (FUSEOPT + 5)   //dequant+silu for ffn_gate to ffn_up hadamard product
#define FUSE_DEQUANT_HADAMARD_QUANTSTAGE1       (FUSEOPT + 6)   //dequant+hadamard+get abs max for smoothquant ffn_up to ffn_down
#define FUSE_DEQUANT_RESADD                     (FUSEOPT + 7)   //dequant+resadd for ffn_down to output

#define SCALE_TYPE_NONE 0
#define SCALE_TYPE_PERTOKEN_A_PERTENSOR_B 1

#include <stdint.h> 
#define Tensor_M 64

#define SEQ_LEN 128
#define RMS_EPSILON 9.999999747378752e-06
#define EMBEDING_DIMENSION 2048
#define KEY_DIMENSION 64
#define VALUE_DIMENSION 64
#define N_HEAD_Q 32
#define N_HEAD_KV 8
#define MAX_CTX_LEN 8192
#define FFN_DIMENSION 8192


static float rope_theta[KEY_DIMENSION/2] __attribute__((aligned(64))) = {1.0000e+00, 6.6360e-01, 4.4037e-01, 2.9223e-01, 1.9392e-01, 1.2869e-01,
        8.5397e-02, 5.6670e-02, 3.7606e-02, 2.4955e-02, 1.6560e-02, 1.0990e-02,
        7.2927e-03, 4.8394e-03, 3.2114e-03, 1.6846e-03, 7.7941e-04, 2.8119e-04,
        8.8651e-05, 5.2790e-05, 3.1436e-05, 1.8720e-05, 1.1147e-05, 6.6380e-06,
        3.9528e-06, 2.3539e-06, 1.4017e-06, 8.3469e-07, 4.9704e-07, 2.9598e-07,
        1.7625e-07, 1.0496e-07};

static float identity[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(256))) = {0};
static float attn_norm_weight[EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t proj_q_weight[N_HEAD_Q][KEY_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_q_scale[1] = {0};
static int8_t proj_k_weight[N_HEAD_KV][KEY_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_k_scale[1] = {0};
static int8_t proj_v_weight[N_HEAD_KV][VALUE_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_v_scale[1] = {0};
static int8_t proj_o_weight[EMBEDING_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_o_scale[1] = {0};
static float  ffn_norm_weight[EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static int8_t ffn_gate_weight[FFN_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_gate_scale[1] = {0};
static int8_t ffn_up_weight[FFN_DIMENSION][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_up_scale[1] = {0};
static int8_t ffn_down_weight[EMBEDING_DIMENSION][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static float  ffn_down_scale[1] = {0};

static int8_t hidden_states_buf_q8_after_pre_rmsnorm[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  hidden_states_buf_q8_after_pre_rmsnorm_scale[SEQ_LEN] = {0};

static float  proj_q_buf_q16[SEQ_LEN][N_HEAD_Q][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_k_buf_q16[SEQ_LEN][N_HEAD_KV][KEY_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_v_buf_q16[SEQ_LEN][N_HEAD_KV][VALUE_DIMENSION] __attribute__((aligned(64))) = {0};

static float  scores_buf_q16[N_HEAD_Q][SEQ_LEN][SEQ_LEN] __attribute__((aligned(64))) = {0};

static int8_t attn_buf_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  attn_buf_q8_scale[SEQ_LEN] = {0};

static int8_t proj_o_buf_after_RMSNORM_q8[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};
static float  proj_o_buf_after_RMSNORM_q8_scale[SEQ_LEN] = {0};

static float ffn_gate_buf_f32[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};

static float ffn_up_buf_f32[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};

static int8_t ffn_up_buf_q8[SEQ_LEN][FFN_DIMENSION] __attribute__((aligned(64))) = {0};
static float ffn_up_buf_q8_scale[SEQ_LEN] __attribute__((aligned(64))) = {0};

static float hidden_states_output[SEQ_LEN][EMBEDING_DIMENSION] __attribute__((aligned(64))) = {0};




#include <math.h>

char *activation_name(int after_ops) {
  switch (after_ops) {
    case NO_ACTIVATION:
      return "NO_ACTIVATION";
    case FUSE_DEQUANT_ROPE_BF16CVRT:
      return "FUSE_DEQUANT_ROPE_BF16CVRT";
    case FUSE_DEQUANT_BF16CVRT:
        return "FUSE_DEQUANT_BF16CVRT";
    case FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT:
        return "FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT";
    case FUSE_DEQUANT_RESADD_RMSNORM_QUANT:
        return "FUSE_DEQUANT_RESADD_RMSNORM_QUANT";
    case FUSE_DEQUANT_SILU:
        return "FUSE_DEQUANT_SILU";
    case FUSE_DEQUANT_HADAMARD_QUANTSTAGE1:
        return "FUSE_DEQUANT_HADAMARD_QUANTSTAGE1";
    case FUSE_DEQUANT_RESADD:
        return "FUSE_DEQUANT_RESADD";
    case PER_TOKEN_QUANT:
        return "PER_TOKEN_QUANT";
    case QUANTSTAGE1:
        return "QUANTSTAGE1";
    default:
      return "UNKNOWN";
  }
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

int cute_buf_id = 0;
int CUTE_result_index = 0;
void * CUTE_result[4] = {(void *) (0x70200000), (void *) (0x70200000 + SEQ_LEN * Tensor_M * 4), (void *) (0x70200000 + SEQ_LEN * Tensor_M * 4 * 2), (void *) (0x70200000 + SEQ_LEN * Tensor_M * 4 * 3)};//double buffer use shuttle tcm
void * CUTE_result_layerwise[2] = {(void *) (0x70200000), (void *) (0x70200000 + Tensor_M * EMBEDING_DIMENSION * 4)};//double buffer use shuttle tcm
void * TCM_BUFF = (void *) (0x70200000);//2MB
float_t quant_absmax_buff[SEQ_LEN]__attribute__((aligned(256))) = {0};

static void matmul_cute(size_t DIM_M, size_t DIM_N, size_t DIM_K,
        const void* A, const void* B, void* C,
        float_t* A_scale_factor, float_t* B_scale_factor,int scale_type,
        size_t stride_A, size_t stride_B, size_t stride_C,
        int datatype,int after_ops,int transpose_result)
{

  if(!(DIM_M % 64 == 0 && DIM_N % 64 == 0 && DIM_K % 64 == 0))
  {
    printf("Can't Till Now!");
    exit(1);
  }

  void (*afater_operation)(void *,int,int,void *,float,int) = NULL;

  switch (after_ops) {
    case NO_ACTIVATION:
      afater_operation = NULL;
      break;
    case FUSE_DEQUANT_ROPE_BF16CVRT://TODO:这里的rope刚好维度是64，所以可以展开
      afater_operation = NULL;
    case FUSE_DEQUANT_BF16CVRT:
      afater_operation = NULL;
    case FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT:
      afater_operation = NULL;
    case FUSE_DEQUANT_RESADD_RMSNORM_QUANT:
      afater_operation = NULL;
    case FUSE_DEQUANT_SILU:
      afater_operation = NULL;
    case FUSE_DEQUANT_HADAMARD_QUANTSTAGE1:
      afater_operation = NULL;
    case FUSE_DEQUANT_RESADD:
      afater_operation = NULL;
    default:
      afater_operation = NULL;
      break;
  }


  if(after_ops != FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT && after_ops != FUSE_DEQUANT_RESADD_RMSNORM_QUANT)
  {
    int Tile_I = DIM_M / 64;
    int Tile_J = DIM_N / 64;

    int Application_M = 64;
    int Application_N = 64;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = Tensor_M * 4;
    int Application_stride_D = 0;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = 0;
    int Is_Zero_Load = 1;
    uint64_t bias_type = TaskTypeTensorZeroLoad;


    uint64_t wait_after_operation_cute_task_id = 0;
    uint64_t wait_after_operation_cute_task_id_pre = 0;

    const void* Tile_A = A;
    const void* Tile_B = B;
    void* Tile_C = CUTE_result[CUTE_result_index];
    void* Tile_D = NULL;

    wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);

    int i = 0;
    int j = 1;
    int pre_i = 0;
    int pre_j = 0;

    int acc_not_finish = 1;
    int next_CUTE_result_index = CUTE_result_index==3?0:CUTE_result_index+1;
    volatile int acc_finish = 0;
    for (i=0;i<Tile_I;i++)
    for (j=(i==0?1:0);j<Tile_J;j++)
    {

        CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
        next_CUTE_result_index = CUTE_result_index==3?0:CUTE_result_index+1;

        Tile_A = A + i * 64 * stride_A;
        Tile_B = B + j * 64 * stride_B;
        Tile_C = CUTE_result[next_CUTE_result_index];//下一组任务
        Tile_D = NULL;

        wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);
        
        afater_operation(CUTE_result[CUTE_result_index],64,64,(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64),0,stride_C);

        CUTE_result_index = next_CUTE_result_index;
        pre_i = i;
        pre_j = j;
    }
    CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
    afater_operation(CUTE_result[CUTE_result_index],64,64,(C+(transpose_result ? pre_j : pre_i)*64*stride_C+(transpose_result ? pre_i : pre_j)*64),0,stride_C);

  }else
  {
    ////TODO:will get proj_o_buf_after_RMSNORM_q8_scale
    int Tile_I = DIM_M / 64;
    // int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = DIM_N;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = EMBEDING_DIMENSION * 4;
    int Application_stride_D = 0;

    int Is_Transpose = 0;

    uint64_t bias_type = TaskTypeTensorZeroLoad;


    uint64_t wait_after_operation_cute_task_id_pre = 0;

    const void* Tile_A = A;
    const void* Tile_B = B;
    void* Tile_C = CUTE_result_layerwise[CUTE_result_index];
    void* Tile_D = NULL;

    wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, datatype, bias_type, Is_Transpose, 0);

    int i = 1;
    int pre_i = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
    for (i=1;i<Tile_I;i++)
    {

        CUTE_TASK_END(wait_after_operation_cute_task_id_pre);

        Tile_A = A + i * 64 * stride_A;
        Tile_B = B;
        Tile_C = CUTE_result_layerwise[CUTE_result_index==0?1:0];
        Tile_D = NULL;
        wait_after_operation_cute_task_id_pre = issue_cute_matmul_marco_inst(Tile_A, Application_stride_A, Tile_B, Application_stride_B, Tile_D, Application_stride_D, Tile_C, Application_stride_C, Application_M, Application_N, Application_K, 1, bias_type, Is_Transpose, 0);

        afater_operation(CUTE_result_layerwise[CUTE_result_index],64,DIM_N,(C+pre_i*64*stride_C),0.0,stride_C);
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }
    CUTE_TASK_END(wait_after_operation_cute_task_id_pre);
    afater_operation(CUTE_result_layerwise[CUTE_result_index],64,DIM_N,(C+pre_i*64*stride_C),0.0,stride_C);
    
  }
}


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


static void matmul_cpu(size_t DIM_I, size_t DIM_J, size_t DIM_K,
        float *A, float *B, float *D, float *C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int act, bool repeating_bias, int transpose_result) {

    return;
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


static float f32_absmax_buf[SEQ_LEN] __attribute__((aligned(64))) = {0};

void smoothquant(float *input, int dim_i, int dim_j, int8_t *output, float_t* output_scale,bool need_stage1) 
{

    
    assert(dim_j%(64*4) == 0);
    assert(dim_i%16 == 0);
    
    if(need_stage1)
    {
        //先对A进行abs_max的量化
        //量化激活
        for (int i = 0; i < dim_i; i++) {
            float* row_A = &input[i * dim_j];
            int8_t* q_row_A = &output[i * dim_j];

            
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(dim_j);
            vl = vl_0;
            vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
                tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
            }
            vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
            tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);

            float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);

            const float d = token_max / (127.0f);
            const float id = d ? 1.0f / d : 0.0f;
            output_scale[i] = d;
            // 第三步，量化
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
                vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
                vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
                __riscv_vse8_v_i8m1(&q_row_A[j], vs, vl);
            }
        }
    }
    else
    {
        //量化激活
        for (int i = 0; i < dim_i; i++) {
            float* row_A = &input[i * dim_j];
            int8_t* q_row_A = &output[i * dim_j];

            const float d = output_scale[i];
            const float id = d ? 1.0f / d : 0.0f;
            // 第三步，量化
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(dim_j);
            vl = vl_0;
            for (int j = 0, avl = dim_j; avl > 0; j += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
                vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
                vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);//默认舍入模式为round to nearest, ties to even
                vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
                __riscv_vse8_v_i8m1(&q_row_A[j], vs, vl);
            }
        }
    }

}

void RMSnorm(float* input, float* output, float* per_channle_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    assert(hidden_dim%(64*4) == 0);
    assert(seq_len%16 == 0);
    
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            // 计算平方和
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0 / sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            // 归一化并缩放
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channle_scale_vec = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channle_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
            }

        }
    }
}

void RMSnorm_With_getabsmax_scale(float* input, float* output, float* per_channle_scale,float* per_token_scale, float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            // 计算平方和
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0 / sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(0.0, vl_0);
            // 归一化并缩放
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channle_scale_vec = __riscv_vle32_v_f32m4(&per_channle_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channle_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
                vfloat32m4_t abs_max_vec = __riscv_vfabs_v_f32m4(scaled_vec, vl);
                max_vec = __riscv_vfmax_vv_f32m4(max_vec, abs_max_vec, vl);
            }

            float token_max = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            per_token_scale[b * seq_len + j] = token_max / (127.0f);
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
    printf("[WorkLoad-(%5d,%5d,*****)LayerWise]RMSnorm_input\n",SEQ_LEN, EMBEDING_DIMENSION);

    RMSnorm_With_Max(identity, TCM_BUFF, attn_norm_weight, hidden_states_buf_q8_after_pre_rmsnorm_scale, RMS_EPSILON, 1, SEQ_LEN, EMBEDING_DIMENSION);
    smoothquant(TCM_BUFF,SEQ_LEN, EMBEDING_DIMENSION,hidden_states_buf_q8_after_pre_rmsnorm, hidden_states_buf_q8_after_pre_rmsnorm_scale, false);

    //proj_q
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_q_weight, proj_q_buf_q16,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_q_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_ROPE_BF16CVRT,0);
    //proj_k
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION / 4, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_k_weight, proj_k_buf_q16,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_k_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION / 4 * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_ROPE_BF16CVRT,0);
    //proj_v
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION / 4, EMBEDING_DIMENSION,
        hidden_states_buf_q8_after_pre_rmsnorm, proj_v_weight, proj_v_buf_q16,
        hidden_states_buf_q8_after_pre_rmsnorm_scale,proj_v_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION / 4 * 2,//bf16
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_BF16CVRT,1);


    //scores
    for (int i = 0; i < N_HEAD_Q; i++) {
        void *A = (void*)proj_q_buf_q16 + i * KEY_DIMENSION * 2;//*2 for bf16 2Byte 
        void *B = (void*)proj_k_buf_q16 + (i/(N_HEAD_Q/N_HEAD_KV)) * KEY_DIMENSION * 2;
        void *C = (void*)scores_buf_q16 + i * SEQ_LEN * SEQ_LEN * 2;

        int factor = 8;//TODO:!KVSCALE
        matmul_cute(SEQ_LEN, SEQ_LEN, KEY_DIMENSION,
            A, B, C,
            NULL,NULL,SCALE_TYPE_NONE,
            KEY_DIMENSION*N_HEAD_Q * 2, KEY_DIMENSION*N_HEAD_KV * 2, SEQ_LEN*2,
            CUTEDataTypeBF16BF16F32,FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT,0);
    }

    //attention
    for (int i = 0; i < N_HEAD_Q; i++) {
        float *A = (void*)scores_buf_q16 + i * SEQ_LEN * SEQ_LEN * 2;//32*128*128
        float *B = (void*)proj_v_buf_q16 + (i/(N_HEAD_Q/N_HEAD_KV)) * SEQ_LEN * VALUE_DIMENSION * 2;//8*64*128
        float *C = (void*)CUTE_result_layerwise + i * VALUE_DIMENSION * 4;//128*2048
        matmul_cute(SEQ_LEN, VALUE_DIMENSION, SEQ_LEN,
            A, B, C,
            NULL,NULL,SCALE_TYPE_NONE,
            SEQ_LEN * 2, SEQ_LEN * 2, EMBEDING_DIMENSION*4,//
            CUTEDataTypeBF16BF16F32,NO_ACTIVATION,0);
    }

    //smoothquant attention
    smoothquant(CUTE_result_layerwise,SEQ_LEN, EMBEDING_DIMENSION,attn_buf_q8,attn_buf_q8_scale, true);


    //proj_o
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, EMBEDING_DIMENSION,
        attn_buf_q8, proj_o_weight, proj_o_buf_after_RMSNORM_q8,
        attn_buf_q8_scale,proj_o_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, EMBEDING_DIMENSION,//int8
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_RESADD_RMSNORM_QUANT,0);//TODO:will get proj_o_buf_after_RMSNORM_q8_scale

    //ffn_gate
    matmul_cute(SEQ_LEN, FFN_DIMENSION, EMBEDING_DIMENSION,
        proj_o_buf_after_RMSNORM_q8, ffn_gate_weight, ffn_gate_buf_f32,
        proj_o_buf_after_RMSNORM_q8_scale,ffn_gate_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, FFN_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_SILU,0);

    //ffn_up
    matmul_cute(SEQ_LEN, FFN_DIMENSION, EMBEDING_DIMENSION,
        proj_o_buf_after_RMSNORM_q8, ffn_up_weight, ffn_up_buf_f32,
        proj_o_buf_after_RMSNORM_q8_scale,ffn_up_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, EMBEDING_DIMENSION, FFN_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_HADAMARD_QUANTSTAGE1,0);//TODO:will get ffn_up_buf_f32_absmax

    smoothquant(ffn_up_buf_f32,SEQ_LEN, FFN_DIMENSION,ffn_up_buf_q8, ffn_up_buf_q8_scale, false);
    //ffn_down
    matmul_cute(SEQ_LEN, EMBEDING_DIMENSION, FFN_DIMENSION,
        ffn_up_buf_f32, ffn_down_weight, hidden_states_output,
        ffn_up_buf_q8_scale,ffn_down_scale,SCALE_TYPE_PERTOKEN_A_PERTENSOR_B,
        EMBEDING_DIMENSION, FFN_DIMENSION, EMBEDING_DIMENSION * 4,//fp32
        CUTEDataTypeI8I8I32,FUSE_DEQUANT_HADAMARD_QUANTSTAGE1,0);


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

    PRINT_LLAMA(/*input_size=*/SEQ_LEN, /*d=*/EMBEDING_DIMENSION, /*dk=*/KEY_DIMENSION, /*dv=*/VALUE_DIMENSION, /*head_q=*/N_HEAD_Q, /*head_kv=*/N_HEAD_KV, /*dffn=*/FFN_DIMENSION);

}

