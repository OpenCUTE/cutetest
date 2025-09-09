### 检查硬件的正确性

本文件夹中的`compare_result.py`为检查软件在处理器上运行正确性的例子

硬件中有trace输出来用于正确性测试和debug,这些trace一般是某些硬件信号在其有效的周期里被输出出来，每个模块都有都有自己的输出信息开关来控制是否输出这些信息，这些选项可以在`chipyard/generators/cute/src/main/scala/CUTEParameters.scala`中设置

带协处理器的加速器运行时的硬件配置在`chipyard/generators/chipyard/src/main/scala/config/CuteConfig.scala`，如

```
class VerifyL2DramPerformenceTest1CUTEM256Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams.copy(Debug = CuteDebugParams.CMLDebugEnable)) ++
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 256) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)
```

中`new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams.copy(Debug = CuteDebugParams.CMLDebugEnable))`设置了cute的参数，这些参数在`chipyard/generators/cute/src/main/scala/CUTEParameters.scala`里的
```
object CuteParams {

    // baseParams:
    def baseParams = CuteParams()

    // 256 bit outside memory bus,128 memory bus
    def TL256Params = baseParams.copy(
        outsideDataWidth = 256,
        MemoryDataWidth = 128
    )

    //default simple debug
    def simpleDebugParams = baseParams.copy(
        Debug = CuteDebugParams.CMLDebugEnable
    )

    //dram&L2 performance test
    def dram_L2_8Tops_PerformanceTestParams = baseParams.copy(
        outsideDataWidth = 512,
        LLCSourceMaxNum = 64,
        MemorysourceMaxNum = 64,
        Tensor_M = 512,
        Tensor_N = 512,
        Tensor_K = 64,
        Matrix_M = 8,
        Matrix_N = 8,
        ReduceWidthByte = 32,
        // Debug = CuteDebugParams.AMLDebugEnable
    )

}
```

里的对应设置定义，可以修改或增添新的设置，控制debug信息是否开启的变量由配置里的`Debug`里的变量控制，这些Debug配置在
```
object CuteDebugParams {

  // NoDebugParams:
  def NoDebug = CuteDebugParams()

  def AMLDebugEnable = NoDebug.copy(
    YJPAMLDebugEnable = true,
  )

  def CMLDebugEnable = NoDebug.copy(
    YJPCMLDebugEnable = true,
  )

  def AllDebugOn = NoDebug.copy(
    YJPDebugEnable = true,
    YJPADCDebugEnable = true,
    YJPBDCDebugEnable = true,
    YJPCDCDebugEnable = true,
    YJPAMLDebugEnable = true,
    YJPBMLDebugEnable = true,
    YJPCMLDebugEnable = true,
    YJPTASKDebugEnable = true,
    YJPVECDebugEnable = true,
    YJPMACDebugEnable = true,
    YJPPEDebugEnable = true,
    YJPAfterOpsDebugEnable = true)
}
```

里定义，如`CMLDebugEnable`里控制的`YJPCMLDebugEnable`可以用来控制CML部件trace信息的开启，这个部件是测试cute正确性最重要的部件，因为它控制了cute最终像访存系统写回结果的过程，检测写回trace的地址和数据是否符合软件运行的预期可以用来测试cute本身的正确性，如果有bug可以再打开其他部件的trace控制变量来对各部件的行为进行进一步检测

* 编写测试脚本检查正确性

用仿真器运行程序得到.out trace文件之后就可以写测试脚本检查正确性，`compare_result.py`是一个检查经过一次macro就完成的矩阵乘是否正确的例子，脚本大致可以分为这几步：

1. 读取trace文件，在这个例子里为

```
# 输入文件名
root_dir = "../../../../" # chipyard路径
config="VerifyL2DramPerformenceTest1CUTEM256Config"
test_name = "cute_Matmul_mnk_512_512_2048_zeroinit_transpose"
input_file = f"{root_dir}sims/verilator/output/chipyard.harness.TestHarness.{config}/{test_name}.out"  # 要筛选的文件
output_file = f"{root_dir}generators/cute/cutetest/matmul/CML_Store_trace.out"  # 保存筛选结果的文件

# 打开输入文件读取，输出文件写入
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    # 遍历输入文件的每一行
    for line in infile:
        # 检查行是否以 "[CML<" 开头
        if line.startswith("[CMemoryLoader_Store<"):
            # 将符合条件的行写入输出文件
            outfile.write(line)

print(f"筛选完成，结果已保存到 {output_file}")
```

`input_file`指明了trace文件的路径，对应的硬件配置名和测试程序名称都需要改成实际执行的情况

这里不仅读入了trace文件，还对CML部件trace的前缀进行了一次筛选，方便后续进一步筛选特定信号时不会与其他部件相混淆

进行多个文件的检测时可用循环包起来并在每次迭代时生成对应的`config`和`test_name`

2. 生成golden序列

通过部件的任务开始提醒或直接生成一个golden序列来算出程序的预期行为，需要读入整体的初始值算出整体的预期值或直接读入预期值，这个例子里.h文件里已有golden,在生成测试初始值时顺带算出

```
# 读取matmul_value_mnk_512_512_2048_zeroinit_transpose.h文件的所有内容
with open(f"./matmul_value_mnk_512_512_2048_zeroinit_transpose.h", "r") as f:
    content = f.read()

    data = content.split("static int gloden_c[512][512] __attribute__((aligned(256))) =")[1].split(";")[0]
    Q_buf = eval(data.replace("{", "[").replace("}", "]"))
```

然后根据对程序和硬件的理解算出全部的结果生成的顺序和对应的写回地址，这个例子里为

```
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
                store_request.append(((currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) 
                                     , (currentstore_blocktensor_reduce_dim_iter + currentstore_blocktensor_subreduce_dim_iter)))
                currentstore_blocktensor_submajor_dim_iter += 1
                if currentstore_blocktensor_submajor_dim_iter >= max_submajor or (currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) >= scp_m:
                    currentstore_blocktensor_submajor_dim_iter = 0
                    currentstore_blocktensor_reduce_dim_iter += per_store_reduce_dim_iter
                    if currentstore_blocktensor_reduce_dim_iter >= scp_n:
                        currentstore_blocktensor_reduce_dim_iter = 0
                        currentstore_blocktensor_major_dim_iter += Matrix_M
                            
            taskid += 1
```

3. 对比trace信号和golden值

挑选出每条写回信号或其他需要检查的中间过程信号并对比对应顺序的golden值，这个例子里包括写回请求的地址和数据,数据可根据地址在整体预期值里索引得到

```
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
            data_golden = Q_buf[major_index][reduce_index : reduce_index + per_store_reduce_dim_iter]
            if (data_golden != data_trace):
                print(f"request_index{request_index},addr{hex(addr_golden)},data_trace{data_trace},data_golden{data_golden}")
                print(f"total_index{total}")
                exit(0)
        
            request_index += 1
            total += 1
```