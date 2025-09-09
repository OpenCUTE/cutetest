# CUTE-Test ReadMe

本仓库是用于存储CUTE使用的神经网络推理的模型的仓库，由gemmini执行的仓库移植而来。目前移植了resnet50和transfomer。

目前两个硬件加速指令，卷积加速指令和矩阵乘加速指令。
卷积加速指令，要求的参数为：
CUTE_CONV_KERNEL_MarcoTask
(void *A,void *B,void *C,void *D,
int Application_M,int Application_N,int Application_K,
int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,
uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,
int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)

ABCD分别为矩阵A、矩阵B和结果矩阵C，偏置矩阵D的起始地址。要求所有矩阵都是Reduce_DIM_FIRST的
Application_M,Application_N,Application_K代表这条宏指令要执行的MNK的长度
conv_stride是卷积的stride步长
kernel_size是卷积核的大小
kernel_stride是每一个index的卷积核的大小，我们要求卷积核的数据排布是(kh,kw,oc,ic)
stride_A、stride_B,stride_C,stride_D代表各个矩阵Reduce_DIM的长度(多少byte)
transpose_result表示结果是否需要进行转置
conv_oh_index,conv_ow_index代表当前处理的矩阵A的起始地址，落在卷积任务input的哪个index上
conv_oh_max,conv_ow_max与index配合，可以完成padding、stride等操作的加速
void * VectorOp,int VectorInst_Length代表了要融合的向量任务的具体指令和指令块长度。

矩阵乘就是IH=1，IW=M，IC=K，OC=N，KH=1，KW=1，STRIDE=1的卷积
