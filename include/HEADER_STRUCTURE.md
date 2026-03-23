# CUTE Header Files Modular Design

## File Organization

### 1. `datatype.h` - Data Type Definitions
**Purpose:** Centralize all data type definitions and width information

**Contents:**
- Element type enum/macro definitions (0-12)
- Byte width definitions for each type (matching CUTEParameters.scala)
- Helper macros to get width by type
- Type name macros for debugging

**Dependencies:** None (base header)

**Used by:** All operation headers, validation.h

---

### 2. `marcohelper.h` - YGJK Instruction Helpers
**Purpose:** Low-level YGJK instruction macros and FIFO operations

**Contents:**
- YGJK instruction bit manipulation macros
- FIFO status check functions
- Low-level register read/write
- Performance counter access
- Custom instruction execution macros

**Dependencies:** None (hardware interface)

**Used by:** convopt.h, matmulopt.h, transformeropt.h

---

### 3. `validation.h` - Parameter Validation
**Purpose:** Compile-time and runtime parameter validation

**Contents:**
- Element type validation
- MNK dimension validation
- Stride calculation and validation macros
- Tiling dimension validation
- Bias type validation
- Convolution dimension validation

**Dependencies:** datatype.h

**Used by:** convopt.h, matmulopt.h, user code

---

### 4. `matmulopt.h` - Matrix Multiplication Operations
**Purpose:** Matrix multiplication operation interfaces

**Contents:**
- Matmul parameter configuration functions
- Safe matmul wrapper with validation
- Stride calculation helpers
- Common matmul patterns

**Dependencies:** datatype.h, marcohelper.h, validation.h

**Used by:** User test code

---

### 5. `convopt.h` - Convolution Operations
**Purpose:** Convolution operation interfaces

**Contents:**
- Conv parameter configuration functions
- Safe conv wrapper with validation
- Conv-specific stride calculations
- Common conv patterns (ResNet, etc.)

**Dependencies:** datatype.h, marcohelper.h, validation.h

**Used by:** User test code, ResNet tests

---

### 6. `transformeropt.h` - Transformer Operations
**Purpose:** Transformer-specific operations and utilities

**Contents:**
- Attention pattern helpers
- RMSNorm parameters
- SmoothQuant utilities
- Transformer-specific validation
- Common layer patterns (QKV projection, FFN, etc.)

**Dependencies:** datatype.h, marcohelper.h, validation.h, matmulopt.h

**Used by:** BERT, LLaMA test code

---

## Dependency Graph

```
datatype.h (base)
    ↓
marcohelper.h (parallel base)
    ↓
validation.h ← depends on: datatype.h
    ↓
┌───────────────┬───────────────┬──────────────────┐
│  matmulopt.h  │  convopt.h    │ transformeropt.h │
└───────────────┴───────────────┴──────────────────┘
    ↓               ↓                ↓
└─────────────── User Test Code ─────────────────┘
```

---

## Migration Plan

### Current Structure
```
cuteMarcoinstHelper.h (monolithic, ~220 lines)
```

### New Structure
```
datatype.h         (~80 lines)  - Type definitions
marcohelper.h      (~100 lines) - YGJK instructions
validation.h       (~150 lines) - Validation macros
matmulopt.h        (~80 lines)  - Matmul operations
convopt.h          (~60 lines)  - Conv operations
transformeropt.h   (~100 lines) - Transformer utilities
```

### Backward Compatibility
Keep `cuteMarcoinstHelper.h` as a compatibility wrapper:
```c
// cuteMarcoinstHelper.h (backward compatibility)
#include "datatype.h"
#include "marcohelper.h"
#include "validation.h"
#include "matmulopt.h"
#include "convopt.h"
```

---

## Usage Examples

### Old Way (still works)
```c
#include "cuteMarcoinstHelper.h"

issue_cute_matmul_marco_inst(a, stride_a, b, stride_b, ...);
```

### New Way (recommended)
```c
#include "matmulopt.h"

// With automatic validation
safe_issue_cute_matmul(a, stride_a, b, stride_b, ...);

// Or manual control
cute_validate_matmul_params(...);
issue_cute_matmul_marco_inst(...);
```

---

## Benefits

1. **Modularity** - Clear separation of concerns
2. **Maintainability** - Easier to find and fix bugs
3. **Reusability** - Other code can include only what they need
4. **Testing** - Can test each module independently
5. **Documentation** - Each file has a clear purpose
6. **Backward Compatible** - Existing tests still work
