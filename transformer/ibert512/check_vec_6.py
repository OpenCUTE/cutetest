input_file = "/home/yuanbin/merge-test/chipyard/sims/verilator/output/chipyard.harness.TestHarness.CUTEShuttle512D512V512M256S1CoreConfig/ibert_6.out"  # 要筛选的文件
output_file = "/home/yuanbin/merge-test/chipyard/generators/cute/cutetest/transformer/ibert512/vpu_Store_trace.out"  # 保存筛选结果的文件
golden_file = "/home/yuanbin/merge-test/chipyard/generators/cute/cutetest/transformer/sublayer/QKV_trace.txt"
h_file = "/home/yuanbin/chipyard/generators/cute/cutetest/transformer/cutetest/transformer-small.h"

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

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # 遍历输入文件的每一行
    for line in infile:
        # 检查行是否以 "[AML<" 开头
        if line.find("store_req:") != -1:
            # 将符合条件的行写入输出文件
            outfile.write(line)

print(f"筛选完成，结果已保存到 {output_file}")

Q_golden = []
K_golden = []
V_golden = []
QKV_golden = []
attn_golden = []
result_golden = []

with open(golden_file, "r") as f:
# with open(f"../conv_value_mnk_49_128_128_k1_s1_oh7.h", "r") as f:
    # 读取所有内容
    content = f.read()
    data = content.split("resadd2_buf = ")[1].split(";")[0]
    result_golden = eval(data.replace("{", "[").replace("}", "]"))
    


# print(V_golden)
# exit(0)

filter_file = "./vpu_Store_trace.out"


with open(filter_file, "r") as infile:
    store_request = []
    request_index = 0


    taskid = 0
    matrixid = 0
    Matrix_M = 8
    scp_n = 512
    scp_m = 64
    # scp_m = 64
    head_num = 4
    head = 0
    application_m = 128
    application_n = 512
    vpu_store_num = application_m * application_n // 64 
    tile_M = 0
    tile_N = 0
    d_datatype = 1
    per_store_reduce_dim_iter = 32 // d_datatype
    max_subreduce = scp_n // per_store_reduce_dim_iter
    max_store_time = application_m * application_n * d_datatype // 32
    
    # __________
    app_stride = 2048
    submajor_max = 4
    currentstore_blocktensor_submajor_dim_iter = 0
    currentstore_blocktensor_reduce_dim_iter = 0
    
    tensor_block_baseaddr = 0
    base_found = False
    line_index = 0
    iter_n = 0
    iter_m = 0
    # 遍历文件每一行
    for line in infile:
                        
            
        # print(line)
        # print(line.split("addr:")[1])
        # print(line.split("addr:")[1].split(",")[0])
        addr_trace = int(line.split("addr=")[1].split(",")[0], 16)

        if (not base_found and addr_trace >= 0x80000000): 
            tensor_block_baseaddr = addr_trace
            base_found = True

        if (addr_trace >= tensor_block_baseaddr and addr_trace < tensor_block_baseaddr + application_m * application_n):
            line_index += 1
            addr_golden = tensor_block_baseaddr + (iter_m + tile_M * scp_m) * application_n + tile_N * scp_n + iter_n
            # addr_golden = tensor_block_baseaddr + (iter_m + tile_M * scp_m) * application_n + tile_N * scp_n + iter_n + head * application_n // head_num
            if (addr_trace != addr_golden):
                print(f"line_index:{line_index},addr_trace:{hex(addr_trace)},addr_golden:{hex(addr_golden)}")
                print(f"iter_m:{iter_m}, iter_n:{iter_n}, tile_M:{tile_M}, tile_N:{tile_N}")
                exit(0)

            data_trace_hex = line.split("data=")[1].split("\n")[0]
            # print(data_trace_hex)
            data_trace_hex = [data_trace_hex[i:i+2] for i in range(0, len(data_trace_hex), 2)][::-1]
            data_trace = [hex2sint(int(x, 16), 8) for x in data_trace_hex]
            
            data_golden = result_golden[iter_m + tile_M * scp_m][iter_n + tile_N * scp_n : iter_n + tile_N * scp_n + 64]
            # data_golden = result_golden[iter_m + tile_M * scp_m][head * application_n // head_num + iter_n + tile_N * scp_n : head * application_n // head_num + iter_n + tile_N * scp_n + 64]

            if (data_golden != data_trace):
                print(f"line_index{line_index},\naddr{hex(addr_golden)},\ndata_trace {data_trace},\ndata_golden{data_golden}")
                exit(0)

            iter_n += 64
            if (iter_n >= scp_n):
                iter_n = 0
                iter_m += 1
                if (iter_m >= scp_m):
                    iter_m = 0
                    tile_N += 1
                    if (tile_N * scp_n >= application_n):
                        tile_N = 0
                        tile_M += 1

    print(f"line_index{line_index}")
        
    