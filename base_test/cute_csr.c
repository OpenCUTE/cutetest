#include <stdio.h>
// // #include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "conv_value_mnk_196_256_256_k3_s1_oh14.h"

int main(void) {

    uint64_t res1 = 1;
    uint64_t A = input;
    uint64_t A_Stride = APPLICATION_K * sizeof(input[0][0]);
    uint64_t B = weight;
    uint64_t B_Stride = APPLICATION_K * sizeof(weight[0][0]);
    uint64_t C = bias;
    uint64_t C_Stride = APPLICATION_N * sizeof(bias[0]);
    uint64_t D = output;
    uint64_t D_Stride = APPLICATION_N * sizeof(output[0][0]);
    uint64_t element_type = 1;//1byte per input
    uint64_t bias_type = TaskTypeTensorRepeatRowLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;
    uint64_t start_cycle = mrdcycle();

    uint64_t issue_val = issue_cute_conv_marco_inst(A, A_Stride, B, B_Stride, C, C_Stride, D, D_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K,KERNEL_STRIDE, element_type, bias_type, TRANSPOSE_RESULT, CONV_STRIDE,CONV_OH_MAX,CONV_OW_MAX,KERNEL_SIZE,CONV_OH_PER_ADD,CONV_OW_PER_ADD,CONV_OH_INDEX,CONV_OW_INDEX);
    res1 = cute_marco_inst_fifo_valid_search();
    printf("FIFO valid search result: %ld   ", res1);
    res1 = cute_marco_inst_fifo_inst_num_search();
    printf("FIFO instruction number: %ld\n", res1);

    issue_val = issue_cute_conv_marco_inst(A, A_Stride, B, B_Stride, C, C_Stride, D, D_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K,KERNEL_STRIDE, element_type, bias_type, TRANSPOSE_RESULT, CONV_STRIDE,CONV_OH_MAX,CONV_OW_MAX,KERNEL_SIZE,CONV_OH_PER_ADD,CONV_OW_PER_ADD,CONV_OH_INDEX,CONV_OW_INDEX);
    res1 = cute_marco_inst_fifo_valid_search();
    printf("FIFO valid search result: %ld   ", res1);
    res1 = cute_marco_inst_fifo_inst_num_search();
    printf("FIFO instruction number: %ld\n", res1);

    issue_val = issue_cute_conv_marco_inst(A, A_Stride, B, B_Stride, C, C_Stride, D, D_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K,KERNEL_STRIDE, element_type, bias_type, TRANSPOSE_RESULT, CONV_STRIDE,CONV_OH_MAX,CONV_OW_MAX,KERNEL_SIZE,CONV_OH_PER_ADD,CONV_OW_PER_ADD,CONV_OH_INDEX,CONV_OW_INDEX);
    res1 = cute_marco_inst_fifo_valid_search();
    printf("FIFO valid search result: %ld   ", res1);
    res1 = cute_marco_inst_fifo_inst_num_search();
    printf("FIFO instruction number: %ld\n", res1);
 
    printf("CUTE CONV REGISTER CONFIGURATION:\n");
    printf("A Base Addr: 0x%lx, A M Stride: %ld\n", A, A_Stride);
    printf("B Base Addr: 0x%lx, B M Stride: %ld\n", B, B_Stride);
    printf("C Base Addr: 0x%lx, C M Stride: %ld\n", C, C_Stride);
    printf("D Base Addr: 0x%lx, D M Stride: %ld\n", D, D_Stride);

    uint64_t M = APPLICATION_M & 0xFFFF;
    uint64_t N = APPLICATION_N & 0xFFFF;
    uint64_t K = APPLICATION_K & 0xFFFF;
    uint64_t cfgData1 = M | (N << 20) | (K << 40);
    printf("CUTE_MNK_KERNALSTRIDE_CONFIG_FUNCTOPS: 0x%lx, 0x%lx\n", cfgData1, KERNEL_STRIDE);

    element_type = element_type & 0xFF;
    bias_type = bias_type & 0xFF;
    uint64_t transpose_result = TRANSPOSE_RESULT & 0xFF;
    uint64_t conv_stride = CONV_STRIDE & 0xFF;
    uint64_t conv_oh_max = CONV_OH_MAX & 0x7FFF;
    uint64_t conv_ow_max = CONV_OW_MAX & 0x7FFF;
    uint64_t kernel_size = KERNEL_SIZE & 0xF;
    uint64_t conv_oh_per_add = CONV_OH_PER_ADD & 0x7FFF;
    uint64_t conv_ow_per_add = CONV_OW_PER_ADD & 0x7FFF;
    uint64_t conv_oh_index = CONV_OH_INDEX & 0x7FFF;
    uint64_t conv_ow_index = CONV_OW_INDEX & 0x7FFF;
    cfgData1 = element_type | (bias_type << 8) | (transpose_result << 16) | (conv_stride << 24) | (conv_oh_max << 32) | (conv_ow_max << 48);
    uint64_t cfgData2 = kernel_size  | (conv_oh_per_add << 4) | (conv_ow_per_add << 19) | (conv_oh_index << 34) | (conv_ow_index << 49);
    printf("CUTE_CONV_CONFIG_FUNCTOPS: 0x%lx, 0x%lx\n", cfgData1, cfgData2);
    printf("CUTE_ISSUE_MARCO_INST\n");

    res1 = cute_Atensor_config_0_search();
    printf("ATensor Config0: 0x%lx   ", res1);
    res1 = cute_Atensor_config_1_search();
    printf("ATensor Config1: 0x%lx\n", res1);
    res1 = cute_Btensor_config_0_search();
    printf("BTensor Config0: 0x%lx   ", res1);
    res1 = cute_Btensor_config_1_search();
    printf("BTensor Config1: 0x%lx\n", res1);
    res1 = cute_Ctensor_config_0_search();
    printf("CTensor Config0: 0x%lx   ", res1);
    res1 = cute_Ctensor_config_1_search();
    printf("CTensor Config1: 0x%lx\n", res1);
    res1 = cute_Dtensor_config_0_search();
    printf("DTensor Config0: 0x%lx   ", res1);
    res1 = cute_Dtensor_config_1_search();
    printf("DTensor Config1: 0x%lx\n", res1);
    res1 = cute_MNK_kernalstride_config_0_search();
    printf("MNK_KERNALSTRIDE Config0: 0x%lx   ", res1);
    res1 = cute_MNK_kernalstride_config_1_search();
    printf("MNK_KERNALSTRIDE Config1: 0x%lx\n", res1);
    res1 = cute_conv_config_0_search();
    printf("Conv Config0: 0x%lx   ", res1);
    res1 = cute_conv_config_1_search();
    printf("Conv Config1: 0x%lx\n", res1);

    printf("issue_val: %ld\n", issue_val);
    //查询指令FIFO的情况
    
    res1 = cute_marco_inst_fifo_valid_search();
    printf("FIFO valid search result: %ld   ", res1);
    res1 = cute_marco_inst_fifo_inst_num_search();
    printf("FIFO instruction number: %ld\n", res1);
    while(res1)
    {
        res1 = cute_marco_inst_fifo_valid_search();
        printf("FIFO valid search result: %ld   ", res1);
        res1 = cute_marco_inst_fifo_inst_num_search();
        printf("FIFO instruction number: %ld\n", res1);
    }
    uint64_t end_cycle = mrdcycle();
    printf("finish\n");
	YGJK_INS_RRR(res1, 0, 0, 3);
	printf("acc read req: %ld\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 4);
	printf("acc write req: %ld\n", res1);

    printf("Cycles: %ld\n", end_cycle - start_cycle);
    return 0;
    
}
