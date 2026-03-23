#ifndef CUTE_DATATYPE_H
#define CUTE_DATATYPE_H

/**
 * @file datatype.h
 * @brief CUTE data type definitions and width information
 *
 * This header centralizes all data type definitions used by CUTE accelerator.
 * Matches the hardware definitions in CUTEParameters.scala
 *
 * Reference: src/main/scala/CUTEParameters.scala:ElementDataType
 */

// ============================================================================
// Element Type Definitions
// ============================================================================

/**
 * @brief CUTE element type identifiers
 *
 * These match the hardware ElementDataType definitions in CUTEParameters.scala
 * Each type represents the input data format for tensor operations
 */
#define CUTEDataTypeI8I8I32       0   /**< Int8 inputs, Int32 output */
#define CUTEDataTypeF16F16F32     1   /**< FP16 inputs, FP32 output */
#define CUTEDataTypeBF16BF16F32   2   /**< BFloat16 inputs, FP32 output */
#define CUTEDataTypeTF32TF32F32   3   /**< TF32 inputs, FP32 output */
#define CUTEDataTypeI8U8I32       4   /**< Int8 * UInt8, Int32 output */
#define CUTEDataTypeU8I8I32       5   /**< UInt8 * Int8, Int32 output */
#define CUTEDataTypeU8U8I32       6   /**< UInt8 inputs, Int32 output */
#define CUTEDataTypeMXFP8E4M3F32  7   /**< MXFP8 E4M3 format, FP32 output */
#define CUTEDataTypeMXFP8E5M2F32  8   /**< MXFP8 E5M2 format, FP32 output */
#define CUTEDataTypeNVFP4F32      9   /**< NVFP4 format (4-bit), FP32 output */
#define CUTEDataTypeMXFP4F32     10   /**< MXFP4 format (4-bit), FP32 output */
#define CUTEDataTypeE4M3F32      11   /**< FP8 E4M3 format, FP32 output */
#define CUTEDataTypeE5M2F32      12   /**< FP8 E5M2 format, FP32 output */

// ============================================================================
// Element Type Bit Width Definitions
// ============================================================================

/**
 * @brief Bit width for each element type (A and B tensors)
 *
 * These definitions match the hardware ElementDataType bit widths in
 * CUTEParameters.scala
 */
#define CUTE_DATABITWIDTH_I8I8I32        8   // 0: Int8
#define CUTE_DATABITWIDTH_F16F16F32      16  // 1: FP16
#define CUTE_DATABITWIDTH_BF16BF16F32    16  // 2: BF16
#define CUTE_DATABITWIDTH_TF32TF32F32    32  // 3: TF32
#define CUTE_DATABITWIDTH_I8U8I32        8   // 4: I8 * UI8
#define CUTE_DATABITWIDTH_U8I8I32        8   // 5: UI8 * I8
#define CUTE_DATABITWIDTH_U8U8I32        8   // 6: UI8 * UI8
#define CUTE_DATABITWIDTH_MXFP8E4M3F32   8   // 7: MXFP8 E4M3
#define CUTE_DATABITWIDTH_MXFP8E5M2F32   8   // 8: MXFP8 E5M2
#define CUTE_DATABITWIDTH_NVFP4F32       4   // 9: NVFP4 (4-bit)
#define CUTE_DATABITWIDTH_MXFP4F32       4   // 10: MXFP4 (4-bit)
#define CUTE_DATABITWIDTH_E4M3F32        8   // 11: E4M3
#define CUTE_DATABITWIDTH_E5M2F32        8   // 12: E5M2

// Result is always 32 bits
#define CUTE_RESULT_BITWIDTH             32  // FP32/Int32 output

// ============================================================================
// Element Width Query Macros
// ============================================================================

/**
 * @brief Get bit width for A/B tensor elements
 * @param element_type Element type identifier (0-12)
 * @return Bit width
 */
#define CUTE_GET_ADATA_BITWIDTH(element_type) \
    ((element_type) == CUTEDataTypeI8I8I32       ? CUTE_DATABITWIDTH_I8I8I32 : \
     (element_type) == CUTEDataTypeF16F16F32     ? CUTE_DATABITWIDTH_F16F16F32 : \
     (element_type) == CUTEDataTypeBF16BF16F32   ? CUTE_DATABITWIDTH_BF16BF16F32 : \
     (element_type) == CUTEDataTypeTF32TF32F32   ? CUTE_DATABITWIDTH_TF32TF32F32 : \
     (element_type) == CUTEDataTypeI8U8I32       ? CUTE_DATABITWIDTH_I8U8I32 : \
     (element_type) == CUTEDataTypeU8I8I32       ? CUTE_DATABITWIDTH_U8I8I32 : \
     (element_type) == CUTEDataTypeU8U8I32       ? CUTE_DATABITWIDTH_U8U8I32 : \
     (element_type) == CUTEDataTypeMXFP8E4M3F32  ? CUTE_DATABITWIDTH_MXFP8E4M3F32 : \
     (element_type) == CUTEDataTypeMXFP8E5M2F32  ? CUTE_DATABITWIDTH_MXFP8E5M2F32 : \
     (element_type) == CUTEDataTypeNVFP4F32      ? CUTE_DATABITWIDTH_NVFP4F32 : \
     (element_type) == CUTEDataTypeMXFP4F32      ? CUTE_DATABITWIDTH_MXFP4F32 : \
     (element_type) == CUTEDataTypeE4M3F32       ? CUTE_DATABITWIDTH_E4M3F32 : \
     (element_type) == CUTEDataTypeE5M2F32       ? CUTE_DATABITWIDTH_E5M2F32 : \
     0)  // Unknown type

/**
 * @brief Get bit width for C tensor (bias) elements
 * @param element_type Element type identifier (0-12)
 * @return Bit width (same as A/B)
 */
#define CUTE_GET_CDATA_BITWIDTH(element_type) \
    CUTE_GET_ADATA_BITWIDTH(element_type)

/**
 * @brief Get bit width for D tensor (output) elements
 * @return Always 32 bits (FP32/Int32)
 */
#define CUTE_GET_DDATA_BITWIDTH(element_type) \
    (void)(element_type), CUTE_RESULT_BITWIDTH

// ============================================================================
// Type Name Macros (for debugging/logging)
// ============================================================================

#define CUTE_DATATYPE_NAME(element_type) \
    ((element_type) == CUTEDataTypeI8I8I32       ? "I8I8I32" : \
     (element_type) == CUTEDataTypeF16F16F32     ? "F16F16F32" : \
     (element_type) == CUTEDataTypeBF16BF16F32   ? "BF16BF16F32" : \
     (element_type) == CUTEDataTypeTF32TF32F32   ? "TF32TF32F32" : \
     (element_type) == CUTEDataTypeI8U8I32       ? "I8U8I32" : \
     (element_type) == CUTEDataTypeU8I8I32       ? "U8I8I32" : \
     (element_type) == CUTEDataTypeU8U8I32       ? "U8U8I32" : \
     (element_type) == CUTEDataTypeMXFP8E4M3F32  ? "MXFP8E4M3F32" : \
     (element_type) == CUTEDataTypeMXFP8E5M2F32  ? "MXFP8E5M2F32" : \
     (element_type) == CUTEDataTypeNVFP4F32      ? "NVFP4F32" : \
     (element_type) == CUTEDataTypeMXFP4F32      ? "MXFP4F32" : \
     (element_type) == CUTEDataTypeE4M3F32       ? "E4M3F32" : \
     (element_type) == CUTEDataTypeE5M2F32       ? "E5M2F32" : \
     "Unknown")

// ============================================================================
// Bias Type Definitions
// ============================================================================

/**
 * @brief Bias/initialization type for C tensor
 */
#define CUTE_BIAS_TYPE_ZERO_INIT    1  /**< Initialize with zeros */
#define CUTE_BIAS_TYPE_ROW_REPEAT    2  /**< Repeat rows across output */
#define CUTE_BIAS_TYPE_FULL_BIAS     3  /**< Full bias tensor */

#define CUTE_BIAS_TYPE_NAME(bias_type) \
    ((bias_type) == CUTE_BIAS_TYPE_ZERO_INIT ? "ZeroInit" : \
     (bias_type) == CUTE_BIAS_TYPE_ROW_REPEAT ? "RowRepeat" : \
     (bias_type) == CUTE_BIAS_TYPE_FULL_BIAS ? "FullBias" : \
     "Unknown")

// ============================================================================
// Tensor Tile Size Definitions
// ============================================================================

/**
 * @brief Default tensor tile dimensions
 *
 * These match the hardware configuration in CUTEParameters.scala
 * Can be overridden for different configurations (8Tops, 16Tops, 32Tops)
 */
#define CUTE_TENSOR_M_ELEMENT_LENGTH  64
#define CUTE_TENSOR_N_ELEMENT_LENGTH  64
#define CUTE_TENSOR_K_ELEMENT_LENGTH  64

// ============================================================================
// Helper Macros for Stride Calculations
// ============================================================================

/**
 * @brief Calculate stride for A tensor in bytes
 * @param element_type Element type identifier
 * @param K K dimension
 */
#define CUTE_CALC_A_STRIDE(element_type, K) \
    ((K) * (CUTE_GET_ADATA_BITWIDTH(element_type) / 8))

/**
 * @brief Calculate stride for B tensor in bytes
 * @param element_type Element type identifier
 * @param K K dimension
 */
#define CUTE_CALC_B_STRIDE(element_type, K) \
    ((K) * (CUTE_GET_ADATA_BITWIDTH(element_type) / 8))

/**
 * @brief Calculate stride for C tensor (bias) in bytes
 * @param element_type Element type identifier
 * @param N N dimension
 */
#define CUTE_CALC_C_STRIDE(element_type, N) \
    ((N) * (CUTE_GET_CDATA_BITWIDTH(element_type) / 8))

/**
 * @brief Calculate stride for D tensor (output) in bytes
 * @param element_type Element type identifier
 * @param N N dimension
 */
#define CUTE_CALC_D_STRIDE(element_type, N) \
    ((N) * (CUTE_GET_DDATA_BITWIDTH(element_type) / 8))

#endif // CUTE_DATATYPE_H
