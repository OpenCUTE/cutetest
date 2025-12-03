#include <stdint.h> 
#define Tensor_M 64

#define SEQ_LEN 128
#define RMS_EPSILON 9.999999747378752e-06
#define EMBEDING_DIMENSION 2048
#define KEY_DIMENSION 64
#define VALUE_DIMENSION 64
#define SQRT_KEY_DIMENSION 8.0
#define INV_SQRT_KEY_DIMENSION 0.125
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

