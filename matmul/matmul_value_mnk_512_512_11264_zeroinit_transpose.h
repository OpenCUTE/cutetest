#define APPLICATION_M 512
#define APPLICATION_N 512
#define APPLICATION_K 11264
#define BIAS_TYPE 1
//1:zero bias,2:repeat row bias,3:full bias
#define CONV_STRIDE 1
#define KERNEL_SIZE 1
#define KERNEL_STRIDE 0
#define STRIDE_A 11264
#define STRIDE_B 11264
#define STRIDE_C 2048
#define STRIDE_D 2048
#define TRANSPOSE_RESULT 1
#define CONV_OH_INDEX 0
#define CONV_OW_INDEX 0
#define CONV_OH_MAX 1
#define CONV_OW_MAX 512
static char a[512][11264] __attribute__((aligned(256)));
static char b[512][11264] __attribute__((aligned(256)));
static int gloden_c[512][512] __attribute__((aligned(256)));
static int c[512][512] __attribute__((aligned(256)));
static int d[512][512] __attribute__((aligned(256)));