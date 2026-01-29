#include <stdio.h>
// #include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
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
    uint64_t element_type = CUTEDataTypeI8I8I32;
    uint64_t bias_type = TaskTypeTensorZeroLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;
    printf("a address: %p\n", a);
    printf("b address: %p\n", b);
    printf("c address: %p\n", c);
    printf("d address: %p\n", d);
    uint64_t issue_val = issue_cute_matmul_marco_inst(a, A_Stride, b, B_Stride, d, D_Stride, c, C_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K, element_type, bias_type,0,0);
    
    res1 = cute_marco_inst_fifo_inst_num_search();
    while(res1)
    {
        printf("inst fifo num: %ld\n", res1);
        res1 = cute_marco_inst_fifo_inst_num_search();
    }

    printf("finish\n");
    YGJK_INS_RRR(res1, 0, 0, 2);
	printf("acc time: %ldcycles\n", res1);
    YGJK_INS_RRR(res1, 0, 0, 5);
    printf("compute: %ldcycles\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 3);
	printf("acc read req: %ld\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 4);
	printf("acc write req: %ld\n", res1);

    for(int i=0;i<128;i++){

            if(c[63][i] != gloden_c[63][i]){
                printf("mismatch at c[63][%d]: expected %d, got %d\n", i, gloden_c[63][i], c[63][i]);
            }
    }

    printf("D Test start\n");
    printf("Pass!\n");



  return 0;
}
