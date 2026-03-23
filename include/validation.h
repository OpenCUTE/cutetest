#ifndef CUTE_VALIDATION_H
#define CUTE_VALIDATION_H

/**
 * @file validation.h
 * @brief Parameter validation functions for CUTE operations
 *
 * This header provides compile-time and runtime validation functions for
 * CUTE accelerator operations to prevent common bugs.
 */

#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include "datatype.h"

// ============================================================================
// Validation Constant Definitions
// ============================================================================

/** Maximum element type identifier */
#define CUTE_MAX_ELEMENT_TYPE 12

/** Maximum supported dimension (16-bit limit) */
#define CUTE_MAX_DIMENSION 65536

/** Minimum dimension value */
#define CUTE_MIN_DIMENSION 1

/** Preferred K dimension alignment for performance */
#define CUTE_K_ALIGNMENT 32

/** Convolution width alignment requirement (64*4 = 256) */
#define CUTE_CONV_WIDTH_ALIGNMENT 256

/** Convolution height alignment requirement */
#define CUTE_CONV_HEIGHT_ALIGNMENT 16

/** Maximum kernel size */
#define CUTE_MAX_KERNEL_SIZE 16

/** Maximum convolution stride */
#define CUTE_MAX_CONV_STRIDE 4

/** Minimum kernel size */
#define CUTE_MIN_KERNEL_SIZE 1

/** Minimum convolution stride */
#define CUTE_MIN_CONV_STRIDE 1

// ============================================================================
// Element Type Validation
// ============================================================================

/**
 * @brief Validate element type is in valid range [0, CUTE_MAX_ELEMENT_TYPE]
 * @param element_type Element type identifier to validate
 */
static inline void cute_validate_element_type(uint64_t element_type) {
    if (element_type > CUTE_MAX_ELEMENT_TYPE) {
        fprintf(stderr, "ERROR: Invalid element_type %lu (must be 0-%d)\n",
                (unsigned long)element_type, CUTE_MAX_ELEMENT_TYPE);
        fprintf(stderr, "   See datatype.h for valid element types\n");
        assert(0 && "Invalid element type");
    }
}

// ============================================================================
// MNK Dimension Validation
// ============================================================================

/**
 * @brief Validate MNK dimensions are within valid range
 * @param M M dimension (rows of output)
 * @param N N dimension (columns of output)
 * @param K K dimension (reduction dimension/shared dimension)
 */
static inline void cute_validate_mnk(uint64_t M, uint64_t N, uint64_t K) {
    assert(M >= CUTE_MIN_DIMENSION && M <= CUTE_MAX_DIMENSION &&
           "M dimension out of valid range");
    assert(N >= CUTE_MIN_DIMENSION && N <= CUTE_MAX_DIMENSION &&
           "N dimension out of valid range");
    assert(K >= CUTE_MIN_DIMENSION && K <= CUTE_MAX_DIMENSION &&
           "K dimension out of valid range");

    if (K % CUTE_K_ALIGNMENT != 0) {
        fprintf(stderr, "WARNING: K=%lu not multiple of %d (may impact performance)\n",
                (unsigned long)K, CUTE_K_ALIGNMENT);
    }
}

// ============================================================================
// Bias Type Validation
// ============================================================================

/**
 * @brief Validate bias type is in valid range [1, 3]
 * @param bias_type Bias type identifier
 */
static inline void cute_validate_bias_type(uint64_t bias_type) {
    if (bias_type < CUTE_BIAS_TYPE_ZERO_INIT ||
        bias_type > CUTE_BIAS_TYPE_FULL_BIAS) {
        fprintf(stderr, "ERROR: Invalid bias_type %lu (must be 1-3)\n",
                (unsigned long)bias_type);
        fprintf(stderr, "   %d = Zero init (CUTE_BIAS_TYPE_ZERO_INIT)\n",
                CUTE_BIAS_TYPE_ZERO_INIT);
        fprintf(stderr, "   %d = Row repeat (CUTE_BIAS_TYPE_ROW_REPEAT)\n",
                CUTE_BIAS_TYPE_ROW_REPEAT);
        fprintf(stderr, "   %d = Full bias (CUTE_BIAS_TYPE_FULL_BIAS)\n",
                CUTE_BIAS_TYPE_FULL_BIAS);
        assert(0 && "Invalid bias type");
    }
}

// ============================================================================
// Stride Validation
// ============================================================================

/**
 * @brief Validate stride for a specific tensor
 * @param tensor_name Name of tensor (for error reporting)
 * @param element_type Element type identifier
 * @param stride Actual stride value in bytes
 * @param dim Dimension value
 * @param expected_stride Expected stride value in bytes
 */
static inline void cute_validate_stride(const char* tensor_name,
                                       uint64_t element_type,
                                       uint64_t stride,
                                       uint64_t dim,
                                       uint64_t expected_stride) {
    if (stride != expected_stride) {
        fprintf(stderr, "ERROR: %s stride mismatch!\n", tensor_name);
        fprintf(stderr, "   Element type: %lu (%s)\n",
                (unsigned long)element_type,
                CUTE_DATATYPE_NAME(element_type));
        fprintf(stderr, "   Element bit width: %d bits\n",
                CUTE_GET_ADATA_BITWIDTH(element_type));
        fprintf(stderr, "   Dimension: %lu\n", (unsigned long)dim);
        fprintf(stderr, "   Expected stride: %lu bytes (dim * bitwidth/8)\n",
                (unsigned long)expected_stride);
        fprintf(stderr, "   Actual stride: %lu bytes\n",
                (unsigned long)stride);
        assert(0 && "Stride validation failed");
    }
}

/**
 * @brief Validate all strides for matrix multiplication
 * @param element_type Element type identifier
 * @param A_stride A tensor stride (bytes)
 * @param B_stride B tensor stride (bytes)
 * @param C_stride C tensor (bias) stride (bytes)
 * @param D_stride D tensor (output) stride (bytes)
 * @param M M dimension
 * @param N N dimension
 * @param K K dimension
 */
static inline void cute_validate_matmul_strides(uint64_t element_type,
                                               uint64_t A_stride,
                                               uint64_t B_stride,
                                               uint64_t C_stride,
                                               uint64_t D_stride,
                                               uint64_t M,
                                               uint64_t N,
                                               uint64_t K) {
    (void)M; // Unused in stride validation
    uint64_t expected_A = CUTE_CALC_A_STRIDE(element_type, K);
    uint64_t expected_B = CUTE_CALC_B_STRIDE(element_type, K);
    uint64_t expected_C = CUTE_CALC_C_STRIDE(element_type, N);
    uint64_t expected_D = CUTE_CALC_D_STRIDE(element_type, N);

    cute_validate_stride("A", element_type, A_stride, K, expected_A);
    cute_validate_stride("B", element_type, B_stride, K, expected_B);
    cute_validate_stride("C (bias)", element_type, C_stride, N, expected_C);
    cute_validate_stride("D (output)", element_type, D_stride, N, expected_D);
}

/**
 * @brief Comprehensive validation for all matmul parameters
 * @param element_type Element type identifier
 * @param bias_type Bias type identifier
 * @param M M dimension
 * @param N N dimension
 * @param K K dimension
 * @param A_stride A tensor stride
 * @param B_stride B tensor stride
 * @param C_stride C tensor stride
 * @param D_stride D tensor stride
 */
static inline void cute_validate_matmul_params(uint64_t element_type,
                                              uint64_t bias_type,
                                              uint64_t M,
                                              uint64_t N,
                                              uint64_t K,
                                              uint64_t A_stride,
                                              uint64_t B_stride,
                                              uint64_t C_stride,
                                              uint64_t D_stride) {
    cute_validate_element_type(element_type);
    cute_validate_bias_type(bias_type);
    cute_validate_mnk(M, N, K);
    cute_validate_matmul_strides(element_type, A_stride, B_stride,
                                C_stride, D_stride, M, N, K);
}

// ============================================================================
// Tiling Dimension Validation
// ============================================================================

/**
 * @brief Validate tensor dimensions for tiling
 * @param app_dim Application-level dimension
 * @param tile_dim Tile-level dimension
 * @param dim_name Dimension name for error reporting
 */
static inline void cute_validate_tiling(uint64_t app_dim,
                                       uint64_t tile_dim,
                                       const char* dim_name) {
    assert(app_dim >= CUTE_MIN_DIMENSION && "Application dimension must be positive");
    assert(tile_dim >= CUTE_MIN_DIMENSION && "Tile dimension must be positive");

    if (app_dim % tile_dim != 0) {
        fprintf(stderr, "ERROR: %s tiling error!\n", dim_name);
        fprintf(stderr, "   App dim: %lu\n", (unsigned long)app_dim);
        fprintf(stderr, "   Tile dim: %lu\n", (unsigned long)tile_dim);
        fprintf(stderr, "   Remainder: %lu (must be 0)\n",
                (unsigned long)(app_dim % tile_dim));
        assert(0 && "Tiling validation failed");
    }
}

// ============================================================================
// Convolution Dimension Validation
// ============================================================================

/**
 * @brief Validate convolution dimensions
 * @param dim_j Width/channel dimension (must be divisible by CUTE_CONV_WIDTH_ALIGNMENT)
 * @param dim_i Height/batch dimension (must be divisible by CUTE_CONV_HEIGHT_ALIGNMENT)
 *
 * Based on patterns in transformer_test/llama/llama3_1B_1.c:894-895
 */
static inline void cute_validate_conv_dims(uint64_t dim_j, uint64_t dim_i) {
    if (dim_j % CUTE_CONV_WIDTH_ALIGNMENT != 0) {
        fprintf(stderr, "ERROR: Conv width %lu not divisible by %d\n",
                (unsigned long)dim_j, CUTE_CONV_WIDTH_ALIGNMENT);
        assert(0 && "Conv width dimension validation failed");
    }

    if (dim_i % CUTE_CONV_HEIGHT_ALIGNMENT != 0) {
        fprintf(stderr, "ERROR: Conv height %lu not divisible by %d\n",
                (unsigned long)dim_i, CUTE_CONV_HEIGHT_ALIGNMENT);
        assert(0 && "Conv height dimension validation failed");
    }
}

/**
 * @brief Validate convolution parameters
 * @param kernel_size Kernel size (must be in range [CUTE_MIN_KERNEL_SIZE, CUTE_MAX_KERNEL_SIZE])
 * @param conv_stride Convolution stride (must be in range [CUTE_MIN_CONV_STRIDE, CUTE_MAX_CONV_STRIDE])
 * @param pad Padding value
 */
static inline void cute_validate_conv_params(uint64_t kernel_size,
                                            uint64_t conv_stride,
                                            int64_t pad) {
    assert(kernel_size >= CUTE_MIN_KERNEL_SIZE &&
           kernel_size <= CUTE_MAX_KERNEL_SIZE &&
           "Kernel size out of valid range");
    assert(conv_stride >= CUTE_MIN_CONV_STRIDE &&
           conv_stride <= CUTE_MAX_CONV_STRIDE &&
           "Conv stride out of valid range");
    assert(pad >= 0 && "Padding must be non-negative");
}

// ============================================================================
// Debug Print Functions
// ============================================================================

/**
 * @brief Print matmul configuration for debugging
 */
static inline void cute_debug_print_matmul_params(uint64_t element_type,
                                                  uint64_t bias_type,
                                                  uint64_t M,
                                                  uint64_t N,
                                                  uint64_t K,
                                                  uint64_t A_stride,
                                                  uint64_t B_stride,
                                                  uint64_t C_stride,
                                                  uint64_t D_stride) {
    printf("=== Matmul Configuration ===\n");
    printf("  Element Type: %lu (%s)\n",
           (unsigned long)element_type,
           CUTE_DATATYPE_NAME(element_type));
    printf("  Bias Type: %lu (%s)\n",
           (unsigned long)bias_type,
           CUTE_BIAS_TYPE_NAME(bias_type));
    printf("  Dimensions: M=%lu, N=%lu, K=%lu\n",
           (unsigned long)M, (unsigned long)N, (unsigned long)K);
    printf("  Strides: A=%lu, B=%lu, C=%lu, D=%lu bytes\n",
           (unsigned long)A_stride, (unsigned long)B_stride,
           (unsigned long)C_stride, (unsigned long)D_stride);
    printf("  Element Bit Width: %d bits\n",
           CUTE_GET_ADATA_BITWIDTH(element_type));
    printf("==========================\n");
}

/**
 * @brief Check stride without asserting (for testing/debugging)
 * @return 1 if stride is valid, 0 otherwise
 */
static inline int cute_check_stride(const char* tensor_name,
                                   uint64_t element_type,
                                   uint64_t stride,
                                   uint64_t dim,
                                   uint64_t expected_stride) {
    int valid = (stride == expected_stride);
    if (!valid) {
        fprintf(stderr, "CHECK FAILED: %s stride mismatch\n", tensor_name);
        fprintf(stderr, "   Expected: %lu, Actual: %lu\n",
                (unsigned long)expected_stride, (unsigned long)stride);
    }
    return valid;
}

#endif // CUTE_VALIDATION_H
