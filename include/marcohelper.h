#ifndef CUTE_MARCOHELPER_H
#define CUTE_MARCOHELPER_H

/**
 * @file marcohelper.h
 * @brief YGJK instruction macros and low-level hardware interface
 *
 * This header provides the low-level interface to CUTE accelerator hardware
 * through custom RISC-V YGJK instructions.
 */

#include <stdint.h>
#include "ygjk.h"

// ============================================================================
// CUTE Instruction Function Codes
// ============================================================================

/** Base function code for CUTE configuration operations */
#define CUTE_CONFIG_FUNCTOPS 64

/** Issue a configured macro instruction to the accelerator */
#define CUTE_ISSUE_MARCO_INST (CUTE_CONFIG_FUNCTOPS + 0)

/** Configure A tensor parameters */
#define CUTE_ATENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 1)

/** Configure B tensor parameters */
#define CUTE_BTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 2)

/** Configure C tensor (bias) parameters */
#define CUTE_CTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 3)

/** Configure D tensor (output) parameters */
#define CUTE_DTENSOR_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 4)

/** Configure MNK dimensions and kernel stride */
#define CUTE_MNK_KERNALSTRIDE_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 5)

/** Configure convolution/matmul parameters */
#define CUTE_CONV_CONFIG_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 6)

/** Dequeue instruction from FIFO */
#define CUTE_FIFO_DEQUEUE_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 16)

/** Get finish tail FIFO index */
#define CUTE_FIFO_GET_FINISH_TAIL_FIFOINDEX_FUNCTOPS (CUTE_CONFIG_FUNCTOPS + 17)

// ============================================================================
// CUTE Status/Query Function Codes
// ============================================================================

/** Base function code for status/query operations */
#define CUTE_SEARCH_FUNCTOPS 0

/** Check if accelerator is currently running */
#define CUTE_IS_RUNNING_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 1)

/** Get running cycle count */
#define CUTE_RUNNING_CYCLYES_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 2)

/** Get memory load request count */
#define CUTE_MRMORY_LOAD_REQUEST_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 3)

/** Get memory store request count */
#define CUTE_MRMORY_STORE_REQUEST_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 4)

/** Get compute cycle count */
#define CUTE_COMPUTE_CYCLYES_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 5)

/** Check if instruction FIFO is finished */
#define CUTE_FIFO_FINISH_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 6)

/** Check if instruction FIFO is full */
#define CUTE_FIFO_FULL_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 7)

/** Check if instruction FIFO has valid entries */
#define CUTE_FIFO_VALID_SEARCH_FUNCTOPS (CUTE_SEARCH_FUNCTOPS + 8)

// ============================================================================
// Low-Level Configuration Functions
// ============================================================================

/**
 * @brief Configure A tensor base address and stride
 * @param ATensor_Base_Addr Base address of A tensor
 * @param ATensor_M_Stride Stride for A tensor (bytes between rows)
 */
static inline void issue_cute_config_ATensor(uint64_t ATensor_Base_Addr,
                                              uint64_t ATensor_M_Stride) {
    int result;
    YGJK_INS_RRR(result, ATensor_Base_Addr, ATensor_M_Stride,
                 CUTE_ATENSOR_CONFIG_FUNCTOPS);
}

/**
 * @brief Configure B tensor base address and stride
 * @param BTensor_Base_Addr Base address of B tensor
 * @param BTensor_M_Stride Stride for B tensor (bytes between rows)
 */
static inline void issue_cute_config_BTensor(uint64_t BTensor_Base_Addr,
                                              uint64_t BTensor_M_Stride) {
    int result;
    YGJK_INS_RRR(result, BTensor_Base_Addr, BTensor_M_Stride,
                 CUTE_BTENSOR_CONFIG_FUNCTOPS);
}

/**
 * @brief Configure C tensor (bias) base address and stride
 * @param CTensor_Base_Addr Base address of C tensor
 * @param CTensor_M_Stride Stride for C tensor (bytes between rows)
 */
static inline void issue_cute_config_CTensor(uint64_t CTensor_Base_Addr,
                                              uint64_t CTensor_M_Stride) {
    int result;
    YGJK_INS_RRR(result, CTensor_Base_Addr, CTensor_M_Stride,
                 CUTE_CTENSOR_CONFIG_FUNCTOPS);
}

/**
 * @brief Configure D tensor (output) base address and stride
 * @param DTensor_Base_Addr Base address of D tensor
 * @param DTensor_M_Stride Stride for D tensor (bytes between rows)
 */
static inline void issue_cute_config_DTensor(uint64_t DTensor_Base_Addr,
                                              uint64_t DTensor_M_Stride) {
    int result;
    YGJK_INS_RRR(result, DTensor_Base_Addr, DTensor_M_Stride,
                 CUTE_DTENSOR_CONFIG_FUNCTOPS);
}

/**
 * @brief Configure MNK dimensions and kernel stride
 * @param M M dimension (rows of output)
 * @param N N dimension (columns of output)
 * @param K K dimension (reduction dimension)
 * @param kernel_stride Stride to next kernel (for convolution)
 *
 * Note: Dimensions are truncated to 16 bits (0-65535)
 */
static inline void issue_cute_config_MNK_KERNALSTRIDE(uint64_t M, uint64_t N,
                                                        uint64_t K,
                                                        uint64_t kernel_stride) {
    int t;
    M = M & 0xFFFF;
    N = N & 0xFFFF;
    K = K & 0xFFFF;
    uint64_t cfgData1 = M | (N << 20) | (K << 40);
    YGJK_INS_RRR(t, cfgData1, kernel_stride, CUTE_MNK_KERNALSTRIDE_CONFIG_FUNCTOPS);
}

/**
 * @brief Configure convolution/operation parameters
 * @param element_type Element type identifier (see datatype.h)
 * @param bias_type Bias type (1=zero, 2=row repeat, 3=full)
 * @param transpose_result Transpose result flag
 * @param conv_stride Convolution stride
 * @param conv_oh_max Output height max
 * @param conv_ow_max Output width max
 * @param kernel_size Kernel size
 * @param conv_oh_per_add Pre-computed oh per add
 * @param conv_ow_per_add Pre-computed ow per add
 * @param conv_oh_index Current oh index
 * @param conv_ow_index Current ow index
 */
static inline void issue_cute_config_CONV(uint64_t element_type,
                                           uint64_t bias_type,
                                           uint64_t transpose_result,
                                           uint64_t conv_stride,
                                           uint64_t conv_oh_max,
                                           uint64_t conv_ow_max,
                                           uint64_t kernel_size,
                                           uint64_t conv_oh_per_add,
                                           uint64_t conv_ow_per_add,
                                           uint64_t conv_oh_index,
                                           uint64_t conv_ow_index) {
    int t;
    element_type = element_type & 0xFF;
    bias_type = bias_type & 0xFF;
    transpose_result = transpose_result & 0xFF;
    conv_stride = conv_stride & 0xFF;
    conv_oh_max = conv_oh_max & 0x7FFF;
    conv_ow_max = conv_ow_max & 0x7FFF;
    kernel_size = kernel_size & 0xF;
    conv_oh_per_add = conv_oh_per_add & 0x7FFF;
    conv_ow_per_add = conv_ow_per_add & 0x7FFF;
    conv_oh_index = conv_oh_index & 0x7FFF;
    conv_ow_index = conv_ow_index & 0x7FFF;

    uint64_t cfgData1 = element_type | (bias_type << 8) | (transpose_result << 16) |
                        (conv_stride << 24) | (conv_oh_max << 32) | (conv_ow_max << 48);
    uint64_t cfgData2 = kernel_size | (conv_oh_per_add << 4) | (conv_ow_per_add << 19) |
                        (conv_oh_index << 34) | (conv_ow_index << 49);
    YGJK_INS_RRR(t, cfgData1, cfgData2, CUTE_CONV_CONFIG_FUNCTOPS);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Read RISC-V cycle counter
 * @return Current cycle count
 */
static inline uint64_t mrdcycle(void) {
    uint64_t res1 = 0;
    asm volatile("rdcycle %0" : "=r"(res1));
    return res1;
}

/**
 * @brief Issue the configured macro instruction to accelerator
 * @return Status code (1 = success)
 */
static inline uint64_t issue_cute_marco_inst(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_ISSUE_MARCO_INST);
    return res1;
}

// ============================================================================
// FIFO Status Functions
// ============================================================================

/**
 * @brief Check if instruction FIFO has valid entries
 * @return Non-zero if FIFO has valid entries, 0 if empty
 */
static inline uint64_t cute_marco_inst_fifo_valid_search(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_VALID_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Check if instruction FIFO is full
 * @return Non-zero if FIFO is full, 0 otherwise
 */
static inline uint64_t cute_marco_inst_fifo_full_search(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_FULL_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Check if instruction FIFO operations are finished
 * @return Non-zero if finished, 0 if still processing
 */
static inline uint64_t cute_marco_inst_fifo_finish_search(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_FINISH_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Dequeue one instruction from FIFO
 * @return Status code
 */
static inline uint64_t cute_marco_inst_fifo_dequeue(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_DEQUEUE_FUNCTOPS);
    return res1;
}

/**
 * @brief Get finish tail FIFO index
 * @return FIFO index
 */
static inline uint64_t cute_marco_inst_fifo_get_finish_tail_fifoindex(void) {
    uint64_t res1 = 1;
    YGJK_INS_RRR(res1, 0, 0, CUTE_FIFO_GET_FINISH_TAIL_FIFOINDEX_FUNCTOPS);
    return res1;
}

// ============================================================================
// Performance Counter Functions
// ============================================================================

/**
 * @brief Get accelerator running cycle count
 * @return Total running cycles
 */
static inline uint64_t cute_get_running_cycles(void) {
    uint64_t res1 = 0;
    YGJK_INS_RRR(res1, 0, 0, CUTE_RUNNING_CYCLYES_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Get compute cycle count
 * @return Compute cycles
 */
static inline uint64_t cute_get_compute_cycles(void) {
    uint64_t res1 = 0;
    YGJK_INS_RRR(res1, 0, 0, CUTE_COMPUTE_CYCLYES_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Get memory load request count
 * @return Number of load requests
 */
static inline uint64_t cute_get_load_requests(void) {
    uint64_t res1 = 0;
    YGJK_INS_RRR(res1, 0, 0, CUTE_MRMORY_LOAD_REQUEST_SEARCH_FUNCTOPS);
    return res1;
}

/**
 * @brief Get memory store request count
 * @return Number of store requests
 */
static inline uint64_t cute_get_store_requests(void) {
    uint64_t res1 = 0;
    YGJK_INS_RRR(res1, 0, 0, CUTE_MRMORY_STORE_REQUEST_SEARCH_FUNCTOPS);
    return res1;
}

#endif // CUTE_MARCOHELPER_H
