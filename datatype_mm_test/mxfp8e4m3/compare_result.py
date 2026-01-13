import numpy as np

test_id = 1
shape = 64 * 2**test_id
bias_type = 3
bias_type_name = ["error", "zeroinit", "rowrepeat", "fullbias"]

# 输入文件名
root_dir = "../../../../../" # chipyard路径
config="CUTEM2564TCUTEShuttle512D512V512M256S1CoreConfig"
# config="VerifyL2DramPerformenceTest1CUTEM256Config"
test_name = f"cute_Matmul_mxfp8_mnk_{shape}_{shape}_{shape}_{bias_type_name[bias_type]}"
input_file = f"{root_dir}sims/verilator/output/chipyard.harness.TestHarness.{config}/{test_name}.out"  # 要筛选的文件
output_file = f"./CML_Store_trace.out"  # 保存筛选结果的文件

# 打开输入文件读取，输出文件写入
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # 遍历输入文件的每一行
    for line in infile:
        # 检查行是否以 "[CML<" 开头
        if line.startswith("[CMemoryLoader_Store<"):
            # 将符合条件的行写入输出文件
            outfile.write(line)

print(f"筛选完成，结果已保存到 {output_file}")


application_m = shape
application_n = shape

Q_buf = []
golden_base = 0
base_get = False

# 将一个无符号数转换为指定长度有符号整数
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

# 读取matmul_value_mnk_512_512_2048_zeroinit_transpose.h文件的所有内容
with open(f"./matmul_value_mxfp8_mnk_{shape}_{shape}_{shape}_{bias_type_name[bias_type]}.h", "r") as f:
    content = f.read()

    data = content.split(f"static int gloden_c[{application_m}][{application_n}] __attribute__((aligned(256))) =")[1].split(";")[0]
    Q_buf = eval(data.replace("{", "[").replace("}", "]"))

input_file = "./CML_Store_trace.out"

store_request = []
request_index = 0

with open(input_file, "r") as infile:
    taskid = 0
    Matrix_M = 4
    scp_n = 64
    scp_m = 64
    task_m = application_m // scp_m
    task_n = application_n // scp_n
    # scp_m = 64
    d_datatype = 4
    per_store_reduce_dim_iter = 64 // d_datatype
    max_subreduce = 16 * 4 // d_datatype
    max_submajor = Matrix_M
    max_load_scp_time = scp_n * scp_m * d_datatype // 32
    transpose = True
    
    # __________
    app_stride = application_n * 4
    currentstore_blocktensor_major_dim_iter = 0
    currentstore_blocktensor_reduce_dim_iter = 0
    
    tensor_block_baseaddr = 0
    total = 0
    # 遍历文件每一行
    for line in infile:
        if line.find("Store D Tensor Start") != -1:
            print("Store D Tensor Start")
            # if taskid >= 2:
            #     scp_m = 49
            store_request.clear()
            request_index = 0
            
            currentstore_blocktensor_submajor_dim_iter = 0
            currentstore_blocktensor_major_dim_iter = 0
            currentstore_blocktensor_subreduce_dim_iter = 0
            currentstore_blocktensor_reduce_dim_iter = 0
            
            for total_store_size in range(max_load_scp_time):
                store_request.append((( currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) 
                                     , ( currentstore_blocktensor_reduce_dim_iter + currentstore_blocktensor_subreduce_dim_iter)))
                currentstore_blocktensor_submajor_dim_iter += 1
                if currentstore_blocktensor_submajor_dim_iter >= max_submajor or (currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) >= scp_m:
                    currentstore_blocktensor_submajor_dim_iter = 0
                    currentstore_blocktensor_reduce_dim_iter += per_store_reduce_dim_iter
                    if currentstore_blocktensor_reduce_dim_iter >= scp_n:
                        currentstore_blocktensor_reduce_dim_iter = 0
                        currentstore_blocktensor_major_dim_iter += Matrix_M
                            
            taskid += 1
            
            
        if line.find("WriteRequest: RequestVirtualAddr=") != -1:
            addr_trace = int(line.split("RequestVirtualAddr=")[1].split(",")[0], 16)
            if (request_index == 0):
                tensor_block_baseaddr = addr_trace
                if (base_get == False):
                    base_get = True
                    golden_base = addr_trace
            
            (major_index, reduce_index) = store_request[request_index]
            addr_golden = tensor_block_baseaddr + major_index * app_stride + reduce_index * d_datatype
            if (addr_trace != addr_golden):
                print(f"request_index:{request_index},addr_trace:{hex(addr_trace)},addr_golden:{hex(addr_golden)}")
                exit(0)
                
            data_trace_hex = line.split("RequestData:")[1].split("\n")[0]
            # print(data_trace_hex)
            data_trace_hex = [data_trace_hex[i:i+8] for i in range(0, len(data_trace_hex), 8)][::-1]
            data_trace = [hex2sint(int(x, 16), 32) for x in data_trace_hex]
            offset = int(addr_golden - golden_base) // 4
            # print(offset)
            data_golden = Q_buf[offset // application_n][offset % application_n : offset % application_n + per_store_reduce_dim_iter]
            if (data_golden != data_trace):
                print(f"request_index{request_index},addr{hex(addr_golden)},data_trace{data_trace},data_golden{data_golden}")
                print(f"total_index{total}")
                exit(0)
        
            request_index += 1
            total += 1
            

        
print("Down!")