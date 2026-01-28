#include <stdio.h>
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "matmul_value_mnk_128_128_128_zeroinit.h"

int main(void) {


    uint64_t res1 = 1;
    // uint64_t A = input;
    uint64_t A_Stride = APPLICATION_K * sizeof(a[0][0]);
    // uint64_t B = weight;
    uint64_t B_Stride = APPLICATION_K * sizeof(b[0][0]);
    // uint64_t C = bias;
    uint64_t C_Stride = APPLICATION_N * sizeof(c[0][0]);
    // uint64_t D = output;
    uint64_t D_Stride = APPLICATION_N * sizeof(d[0][0]);
    uint64_t element_type = 0;//1byte per input
    uint64_t bias_type = TaskTypeTensorZeroLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;

    cute_marco_inst_mmu_flush_usingVM(1, 1);
    for(int i=0;i<APPLICATION_M*APPLICATION_K;i+=4096){
        volatile char tmp = a[i];
        (void)tmp;
    }
    for(int i=0;i<APPLICATION_K*APPLICATION_N;i+=4096){
        volatile char tmp = b[i];
        (void)tmp;
    }
    for(int i=0;i<APPLICATION_M*APPLICATION_N;i+=4096){
        volatile char tmp = c[i];
        (void)tmp;
    }

    uint64_t issue_val = issue_cute_matmul_marco_inst(a, A_Stride, b, B_Stride, d, D_Stride, c, C_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K, element_type, bias_type,0,0);
    
    res1 = cute_marco_inst_fifo_inst_num_search();
    while(res1)
    {
        res1 = cute_marco_inst_fifo_inst_num_search();
    }
    printf("finish\n");

    for(int i=0;i<128;i++){

            if(c[63][i] != gloden_c[63][i]){
                printf("mismatch at c[63][%d]: expected %d, got %d\n", i, gloden_c[63][i], c[63][i]);
            }
    }
    printf("test over\n");
  return 0;
}
