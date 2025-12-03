#define APPLICATION_M 1024
#define APPLICATION_N 1024
#define APPLICATION_K 4864
#define BIAS_TYPE 1
//1:zero bias,2:repeat row bias,3:full bias
#define CONV_STRIDE 1
#define KERNEL_SIZE 1
#define KERNEL_STRIDE 0
#define STRIDE_A 4864
#define STRIDE_B 4864
#define STRIDE_C 8192
#define STRIDE_D 8192
#define TRANSPOSE_RESULT 1
#define CONV_OH_INDEX 0
#define CONV_OW_INDEX 0
#define CONV_OH_MAX 1
#define CONV_OW_MAX 512
static char a[1024][4864] __attribute__((aligned(256)));
static char b[1024][4864] __attribute__((aligned(256)));
static int gloden_c[1024][1024] __attribute__((aligned(256)));
static int c[1024][1024] __attribute__((aligned(256)));
static int d[1024][1024] __attribute__((aligned(256)));