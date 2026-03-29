#include "ygjk.h"
#include <stdint.h>
#include "datatype.h"
#include "instruction.h"
#include "validation.h"

uint64_t mrdcycle()
{
    uint64_t res1=0;
    asm volatile("rdcycle %0":"=r"(res1));
    return res1;
}

// ========================================
// 二次包装函数（手写，不在自动生成范围内）
// ========================================

void issue_cute_config_MatMul(uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t current_M_index)
{
    uint64_t conv_stride = 1;
    uint64_t conv_oh_max = 0;//TODO:这两个为零，会被识别为矩阵乘，太dirty了
    uint64_t conv_ow_max = 0;//TODO:这两个为零，会被识别为矩阵乘，太dirty了
    uint64_t kernel_size = 1;
    uint64_t conv_oh_per_add = 0;
    uint64_t conv_ow_per_add = CUTE_TENSOR_M;//
    uint64_t conv_oh_index = 0;
    uint64_t conv_ow_index = current_M_index;
    CUTE_CONFIG_CONV_PARAMS(element_type, bias_type, transpose_result, conv_stride, conv_oh_max, conv_ow_max,
                             kernel_size, conv_oh_per_add, conv_ow_per_add, conv_oh_index, conv_ow_index);
}

uint64_t issue_cute_conv_marco_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
                                       uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
                                       uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
                                       uint64_t DTensor_Base_Addr,uint64_t DTensor_M_Stride,
                                       uint64_t M,uint64_t N,uint64_t K,uint64_t kernel_stride,
                                       uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t conv_stride,uint64_t conv_oh_max,uint64_t conv_ow_max,
                                       uint64_t kernel_size,uint64_t conv_oh_per_add,uint64_t conv_ow_per_add,uint64_t conv_oh_index,uint64_t conv_ow_index)
{
    CUTE_CONFIG_TENSOR_A(ATensor_Base_Addr,ATensor_M_Stride);
    CUTE_CONFIG_TENSOR_B(BTensor_Base_Addr,BTensor_M_Stride);
    CUTE_CONFIG_TENSOR_C(CTensor_Base_Addr,CTensor_M_Stride);
    CUTE_CONFIG_TENSOR_D(DTensor_Base_Addr,DTensor_M_Stride);
    CUTE_CONFIG_TENSOR_DIM(M,N,K,kernel_stride);
    CUTE_CONFIG_CONV_PARAMS(element_type,bias_type,transpose_result,conv_stride,conv_oh_max,conv_ow_max,
                             kernel_size,conv_oh_per_add,conv_ow_per_add,conv_oh_index,conv_ow_index);
    return CUTE_SEND_MACRO_INST();
}

uint64_t  issue_cute_matmul_marco_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
                                       uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
                                       uint64_t BiasTensor_Base_Addr,uint64_t BiasTensor_M_Stride,
                                       uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
                                       uint64_t M,uint64_t N,uint64_t K,
                                       uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t matmul_m_index)
{
    CUTE_CONFIG_TENSOR_A(ATensor_Base_Addr,ATensor_M_Stride);
    CUTE_CONFIG_TENSOR_B(BTensor_Base_Addr,BTensor_M_Stride);
    CUTE_CONFIG_TENSOR_C(BiasTensor_Base_Addr,BiasTensor_M_Stride);
    CUTE_CONFIG_TENSOR_D(CTensor_Base_Addr,CTensor_M_Stride);
    CUTE_CONFIG_TENSOR_DIM(M,N,K,0);
    issue_cute_config_MatMul(element_type,bias_type,transpose_result,matmul_m_index);
    return CUTE_SEND_MACRO_INST();
}

uint64_t issue_cute_blockscale_matmul_macro_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
                                       uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
                                       uint64_t AScale_Base_Addr, uint64_t BScale_Base_Addr,
                                       uint64_t BiasTensor_Base_Addr,uint64_t BiasTensor_M_Stride,
                                       uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
                                       uint64_t M,uint64_t N,uint64_t K,
                                       uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t matmul_m_index)
{
    CUTE_CONFIG_TENSOR_A(ATensor_Base_Addr,ATensor_M_Stride);
    CUTE_CONFIG_TENSOR_B(BTensor_Base_Addr,BTensor_M_Stride);
    CUTE_CONFIG_SCALE_A(AScale_Base_Addr);
    CUTE_CONFIG_SCALE_B(BScale_Base_Addr);
    CUTE_CONFIG_TENSOR_C(BiasTensor_Base_Addr,BiasTensor_M_Stride);
    CUTE_CONFIG_TENSOR_D(CTensor_Base_Addr,CTensor_M_Stride);
    CUTE_CONFIG_TENSOR_DIM(M,N,K,0);
    issue_cute_config_MatMul(element_type,bias_type,transpose_result,matmul_m_index);
    return CUTE_SEND_MACRO_INST();
}

// CUSTOM1 测试（不在 instruction.h 范围内）
uint64_t cute_marco_inst_tma_test()
{
    uint64_t res1=1;
    YGJK_INS_CUSTOM1_RRR(res1, 0, 0, 0);
    return res1;
}
