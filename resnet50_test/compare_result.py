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
    
def round_near_even_with_scale(x, scale):
    return x >> scale

def after_relu(x, scale):
    x = round_near_even_with_scale(x, scale)
    if x > 127:
        return 127
    if x < -128:
        return -128
    return x
    
config = "CUTEShuttle512D512V512M256S1CoreConfig"
# config = "CUTETestConfig"
root_dir = "/home/yuanbin/merge-test/chipyard/"

for test_index in range(13, 14):
    print(f"golden{test_index} start")
    # 输入文件名
    input_file = f"{root_dir}sims/verilator/output/chipyard.harness.TestHarness.{config}/vec_ops_conv_{test_index}.out"  # 要筛选的文件
    
    app_m = 0
    app_k = 0
    app_n = 0
    out_row = 0
    in_row = 0
    stride = 0
    kernel_size = 0
    output_scale_shift = 0

    store_request = []
    request_index = 0
    
    input = []
    weight = []
    bias = []
    conv_golden = []
    martix_golden_c = []
    
    load_pc = ""
    base_found = False
    output_base = 0
    
    with open(f"./vec_ops_conv_{test_index}.dump", "r") as f:
        for line in f.readlines():
            if line.find("02068107          	vle8.v	v2,(a3)") != -1:
                load_pc = line.split(":")[0].strip()
                print(load_pc)
        
    
    with open(f"./conv_{test_index}.h", "r") as f:
    # with open(f"../conv_value_mnk_49_128_128_k1_s1_oh7.h", "r") as f:
        # 读取所有内容
        content = f.read()
        app_m = int(content.split("#define APPLICATION_M ")[1].split("\n")[0])
        app_k = int(content.split("#define APPLICATION_K ")[1].split("\n")[0])
        app_n = int(content.split("#define APPLICATION_N ")[1].split("\n")[0])
        out_row = int(content.split(".out_row_dim =")[1].split(",")[0])
        stride = int(content.split(".stride =")[1].split(",")[0])
        output_scale_shift = int(content.split(".output_scale_shift = ")[1].split(",")[0])
        in_row = stride * out_row
        kernel_size = int(content.split(".kernel_size =")[1].split(",")[0])
        print(f"app_m:{app_m},app_k:{app_k},app_n:{app_n},out_row:{out_row},stride:{stride},in_row:{in_row},kernel_size:{kernel_size}")
        # 找到static char a[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
        # data = content.split(f"static char input[49][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        data = content.split("int8_t input[A_APPLICATION_M*APPLICATION_K] __attribute__((aligned(64))) = ")[1].split(";")[0]
        input = eval(data.replace("{", "[").replace("}", "]"))
        print(len(input))
        # print(martix_A)
        
        # 找到static char b[128][128] __attribute__((aligned(256))) = 后 ; 之前的内容
        data = content.split("int8_t weights[KERNEL_SIZE*KERNEL_SIZE*APPLICATION_N*APPLICATION_K] __attribute__((aligned(64))) = ")[1].split(";")[0]
        weight = eval(data.replace("{", "[").replace("}", "]"))
        # print(martix_B)
        
        # 找到static int d[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
        # data = content.split("static int bias[128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        data = content.split("int32_t bias[APPLICATION_N] __attribute__((aligned(64))) = ")[1].split(";")[0]
        bias = eval(data.replace("{", "[").replace("}", "]"))
        
        # 找到static int gloden_c[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
        data = content.split(f"int8_t gloden_output_with_scale[APPLICATION_M*APPLICATION_N] __attribute__((aligned(64))) = ")[1].split(";")[0]
        # data = content.split("static int gloden_c[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        martix_golden_c = eval(data.replace("{", "[").replace("}", "]"))

        print(f"result_sum:{sum(martix_golden_c) % 256}")
        
        conv_golden = []
        for i in range(app_m):
            line = []
            for j in range(app_n):
                line.append(0)
            conv_golden.append(line)
            
        for oh in range(0, out_row):
            for ow in range(0, out_row):
                for oc in range(0, app_n):
                    temp_acc = 0
                    for kh in range(-(kernel_size // 2), kernel_size // 2 + 1):
                        for kw in range(-(kernel_size // 2), kernel_size // 2 + 1):
                            ih = oh * stride + kh
                            iw = ow * stride + kw
                            if ih < 0 or ih >= in_row or iw < 0 or iw >= in_row:
                                continue
                            input_index = (ih * in_row + iw) * app_k
                            weight_index = (((kh + kernel_size // 2) * kernel_size + kw + kernel_size // 2) * app_n + oc) * app_k
                            for ic in range(0, app_k):
                                # print(f"ih:{ih},iw:{iw},kh:{kh},kw:{kw},oc:{oc},ic:{ic}")
                                try:
                                    temp_acc += input[input_index + ic] * weight[weight_index + ic]
                                except:
                                    print(f"ih:{ih},iw:{iw},kh:{kh},kw:{kw},oc:{oc},ic:{ic},input_index:{input_index},weight_index:{weight_index}")
                                    exit(0)
                    temp_acc += bias[oc]
                    conv_golden[oh * out_row + ow][oc] = temp_acc
                    

    with open(input_file, "r") as infile:
        sb_addr = dict()
        result_base = 0x80068b00
        
        check_addr_num = 0
        
        false_num = 0
        
        req_index = [0, -2]
        resp_index = (0, -2)
        
        tile_M = 0
        tile_N = 0
        M_sub = 0
        N_Sub = 0
        subreduce = 0
        submajor = 0
        
        cml_write_num = 0
        # 遍历文件每一行带索引
        for index, line in enumerate(infile.readlines()):
            if line.find("WriteRequest: RequestVirtualAddr=") != -1:
                cml_write_num += 1
                data_trace = line.split("RequestData:")[1].split("\n")[0]
                data_trace_hex = [data_trace[i:i+8] for i in range(0, len(data_trace), 8)][::-1]
                data_trace = [hex2sint(int(x, 16), 32) for x in data_trace_hex]
                current_m = tile_M * 64 + M_sub + submajor
                current_n = tile_N * 64 + N_Sub + subreduce
                if data_trace != conv_golden[current_m][current_n : current_n + 16]:
                    print(f"tile_M:{tile_M},tile_N:{tile_N},M_sub:{M_sub},N_sub:{N_Sub}")
                    print(f"read_data:  {data_trace}")
                    print(f"expect_data:{conv_golden[tile_M * 64 + M_sub][tile_N * 64 + N_Sub : tile_N * 64 + N_Sub + 8]}")
                    exit(0)
                
                subreduce += 16
                if subreduce == 16:
                    subreduce = 0
                    submajor += 1
                    if submajor == 4 or tile_M * 64 + M_sub + submajor >= app_m:
                        submajor = 0
                        N_Sub += 16
                        if N_Sub == 64:
                            N_Sub = 0
                            M_sub += 4
                            if M_sub == 64 or tile_M * 64 + M_sub >= app_m:
                                M_sub = 0
                                tile_N += 1
                                if tile_N * 64 >= app_n:
                                    tile_N = 0
                                    tile_M += 1
                                    if tile_M * 64 >= app_m:
                                        print("cml store check end")
                    
                        
            if line.find(f"req.pc = 00{load_pc}") != -1:
                req_index[1] = index
                
            if index == req_index[1] + 2 and line.find("[YJP_VPU_DEBUG_req_seq_id<") != -1:
                req_index[0] = int(line.split("reqHelp.sbId = ")[1].split(",")[0])
                
            if index == req_index[1] + 3 and line.find("[YJP_VPU_DEBUG_req_load<") != -1:
                load_addr = int(line.split("addr: ")[1].split(",")[0], 16)
                sb_addr[req_index[0]] = load_addr
                # print(req_index)
                if base_found == False:
                    output_base = load_addr
                    base_found = True
                
            if base_found and line.find("resp.sbId = ") != -1:
                sbid = int(line.split("resp.sbId = ")[1].split(",")[0])
                if sbid in sb_addr.keys():
                    resp_addr = sb_addr[sbid]
                    if base_found and resp_addr >= output_base and resp_addr < output_base + app_m * app_n:
                        resp_index = (sbid, index)
                
            if index == resp_index[1] + 1 and line.find("resp.s0l1 = 1") != -1:
                check_addr_num += 1
                read_data = line.split("resp.data = ")[1].split("\n")[0]
                offset = sb_addr[resp_index[0]] - output_base
                golden_data = martix_golden_c[offset : offset + 32]
                data_trace_hex = [read_data[i:i+2] for i in range(0, len(read_data), 2)][::-1]
                data_trace = [hex2sint(int(x, 16), 8) for x in data_trace_hex]
                # print(["{:04x}".format(num) for num in golden_data[::-1]])
                golden_str = ["{:04x}".format(num)[2:] for num in golden_data[::-1]]
                golden_str = "".join(golden_str)
                # 将golden_data转换为十六进制数并表示成两位字符串，位数不够填零
                
                if data_trace != golden_data:
                    print(f"sb_id:{resp_index[0]},addr:{hex(sb_addr[resp_index[0]])}")
                    print(f"offset:{offset}")
                    print(f"read_data:  {data_trace}")
                    print(f"expect_data:{golden_data}")
                    print(f"check_addr_num:{check_addr_num}")
                    false_num += 1
                    # exit(0)
        
        # offset = 197600
        # snip = conv_golden[offset // app_n][offset % app_n + 16 : offset % app_n + 24]
        # print(snip)
        # print([hex(x) for x in snip])
        # print([after_relu(x, output_scale_shift) for x in conv_golden[offset // app_n][offset % app_n : offset % app_n + 32]])
        print(f"cml_write_num:{cml_write_num}")
        print(f"check_addr_num:{check_addr_num}")
        print(f"false_num:{false_num}")
        
        output_sum = 0
        for i in range(0, app_m * app_n):
            output_sum += martix_golden_c[i]
        print(f"output_sum:{output_sum}")
                

            
    print("Down!")