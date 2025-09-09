import numpy as np

# 配置名
config = "CUTETestConfig128bitdram512bitL2Widen3issueBoom"
# 测试程序名
test = "transformer_cute"
input_file = f"/home/yuanbin/chipyard/sims/verilator/output/chipyard.TestHarness.{config}/{test}.out"  # 要筛选的trace文件
output_file = "/home/yuanbin/chipyard/generators/cute/cutetest/transformer/sublayer/MTE_trace.out"  # 保存筛选结果的文件

# 将特定部件的trace筛选出来
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # 检查行是否以 "[AML<" 开头
        if line.startswith("[MatrixTE<"):
            outfile.write(line)

print(f"筛选完成，结果已保存到 {output_file}")


martix_A = []
martix_B = []
martix_D = []
martix_golden_c = []

# 将一个指定长度二进制转换为指定长度有符号整数
def hex2sint(num, length):
    # 将num用特定长度length二进制表示出来
    binary = bin(num & int("1"*length, 2))
    # 将binary填充到length长度
    binary = binary[2:].zfill(length)
    # 如果是负数
    if binary[0] == "1":
        return int(binary, 2) - 2**length
    else:
        return int(binary, 2)

# 读取.h文件的所有内容
with open(f"./transformer-small.h", "r") as f:
    # 读取所有内容
    content = f.read()
    data = content.split("static const elem_t Wqkvo[4][512][512] = ")[1].split(";")[0]
    Wqkvo = eval(data.replace("{", "[").replace("}", "]"))
    martix_B = Wqkvo[0]
    # print(martix_A)
    
    # 找到static char b[128][128] __attribute__((aligned(256))) = 后 ; 之前的内容
    data = content.split("static const elem_t input[128][512] = ")[1].split(";")[0]
    martix_A = eval(data.replace("{", "[").replace("}", "]"))
    # print(martix_B)
    
    # 找到static int d[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
    data = content.split("static int d[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
    martix_D = eval(data.replace("{", "[").replace("}", "]"))
    
    # 找到static int gloden_c[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
    data = content.split("static int gloden_c[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
    martix_golden_c = eval(data.replace("{", "[").replace("}", "]"))
    
B = np.array(martix_B)
B = B.T
result = np.dot(martix_A, B) + martix_D
golden_result = np.array(martix_golden_c)
if not np.array_equal(result, golden_result):
    print("Wrong!")
    
ReduceDim = 16
    
# 读取文件MTE_trace.out
with open("./MTE_trace.out", "r") as f:
    vector_A_index = 0
    vector_C_index = 0
    vector_D_index = 0
    
    vector_A_queue = []
    vector_B_queue = []
    vector_C_queue = []
    
    
    # 遍历每一行
    for line in f:
        if line.find("VectorA:") != -1:
            vector_A_hex = line.split("VectorA:")[1].split("\n")[0]
            # print(vector_A_hex)
            # 将字符串vector_A_hex以两个字符为一个整体切分并倒转
            vector_A_hex = [vector_A_hex[i:i+2] for i in range(0, len(vector_A_hex), 2)][::-1]
            # 将字符串组vector_A_hex拼接成一个字符串
            vector_A_hex = "".join(vector_A_hex)
            # print(vector_A_hex)
            # print(vector_A_hex[0:2])
            vector_A = [[hex2sint(int(vector_A_hex[(j * ReduceDim + i) * 2:(j * ReduceDim + i + 1) * 2], 16), 8) for i in range(ReduceDim)] for j in range(4)]
            vector_A_queue.append(vector_A)
            # 计算子矩阵的起始横纵坐标
            sub_index = vector_A_index % 512
            major_index = vector_A_index // 512
            x = major_index % 2 * 64 + sub_index % 2 * 32 
            y = major_index // 4 * 64 + sub_index // 32 * 4
            sub_mat_A = [martix_A[i][x:x+32] for i in range(y, y+4)]
            if vector_A != sub_mat_A:
                print(f"VectorA wrong, VectorAindex{vector_A_index}, vector_A:{vector_A}, sub_mat_A:{sub_mat_A}")
                exit(0)
                
        if line.find("VectorB:") != -1:
            vector_B_hex = line.split("VectorB:")[1].split("\n")[0]
            vector_B_hex = [vector_B_hex[i:i+2] for i in range(0, len(vector_B_hex), 2)][::-1]
            vector_B_hex = "".join(vector_B_hex)
            vector_B = [[hex2sint(int(vector_B_hex[(j * ReduceDim + i) * 2:(j * ReduceDim + i + 1) * 2], 16), 8) for i in range(ReduceDim)] for j in range(4)]
            vector_B_queue.append(vector_B)
            sub_index = vector_A_index % 512
            major_index = vector_A_index // 512
            x = major_index % 2 * 64 + sub_index % 2 * 32 
            y = major_index // 2 % 2 * 64 + sub_index // 2 % 16 * 4
            sub_mat_B = [martix_B[i][x:x+32] for i in range(y, y+4)]
            if vector_B != sub_mat_B:
                print(f"vector_B wrong")
                exit(0)
            
            vector_A_index += 1
            
        if line.find("MatirxC:") != -1:
            matrix_C_hex = line.split("MatirxC:")[1].split("\n")[0]
            matrix_C_hex = [matrix_C_hex[i:i+8] for i in range(0, len(matrix_C_hex), 8)][::-1]
            matrix_C_hex = "".join(matrix_C_hex)
            matrix_C = [[hex2sint(int(matrix_C_hex[(j * 4 + i) * 8:(j * 4 + i) * 8 + 8], 16), 32) for i in range(4)] for j in range(4)]
            vector_C_queue.append(matrix_C)
            sub_index = vector_C_index % 256
            major_index = vector_C_index // 256
            if major_index % 2 == 0:
                x = major_index // 2 % 2 * 64 + sub_index % 16 * 4
                y = major_index // 4 * 64 + sub_index // 16 * 4
                sub_mat_D = [martix_D[i][x:x+4] for i in range(y, y+4)]
                if matrix_C != sub_mat_D:
                    print(f"MatrixC wrong x{x} y{y} MatrixCindex{vector_C_index}, matrix_C:{matrix_C}, sub_mat_D:{sub_mat_D}")
                    exit(0)
                    
            
            vector_C_index += 1
            
        if line.find("MatrixD:") != -1:
            matrix_D_hex = line.split("MatrixD:")[1].split("\n")[0]
            matrix_D_hex = [matrix_D_hex[i:i+8] for i in range(0, len(matrix_D_hex), 8)][::-1]
            matrix_D_hex = "".join(matrix_D_hex)
            matrix_D = [[hex2sint(int(matrix_D_hex[(j * 4 + i) * 8:(j * 4 + i) * 8 + 8], 16), 32) for i in range(4)] for j in range(4)]
            vector_A = vector_A_queue.pop(0)
            vector_B = vector_B_queue.pop(0)
            # 将vector_A和vector_B转换成numpy数组
            vector_A = np.array(vector_A)
            vector_B = np.array(vector_B)
            # vector_B转置
            vector_B = vector_B.T
            # 计算矩阵乘法
            matrix_C = np.dot(vector_A, vector_B)
            # 两次子矩阵乘累加得到结果
            vector_A = vector_A_queue.pop(0)
            vector_B = vector_B_queue.pop(0)
            vector_A = np.array(vector_A)
            vector_B = np.array(vector_B)
            vector_B = vector_B.T
            matrix_C += np.dot(vector_A, vector_B)
            matrix_C += vector_C_queue.pop(0)
            # 将matrix_C转换成int list
            matrix_C = matrix_C.tolist()
            if matrix_C != matrix_D:
                print(f"MatrixD wrong MatrixDindex{vector_D_index}, matrix_D:{matrix_D}, matrix_C:{matrix_C}")
                exit(0)
            sub_index = vector_D_index % 256
            major_index = vector_D_index // 256
            if major_index % 2 == 1:
                x = major_index // 2 % 2 * 64 + sub_index % 16 * 4
                y = major_index // 4 * 64 + sub_index // 16 * 4
                golden_c = [martix_golden_c[i][x:x+4] for i in range(y, y+4)]
                if matrix_C != golden_c:
                    print(f"golden wrong MatrixDindex{vector_D_index}, matrix_C:{matrix_C}, golden_c:{golden_c}")
                    # exit(0)
            vector_D_index += 1
print("Down!")