import subprocess
import os
from typing import Optional

def run_executable(executable_path: str, 
                  input_data: str):
    """
    调用可执行文件并传递输入
    
    参数:
        executable_path: 可执行文件路径
        input_data: 要传递给程序的输入字符串
        timeout: 超时时间(秒)，None表示不限制
    
    返回:
        (return_code, stdout_output, stderr_output)
    """
    # 启动子进程
    process = subprocess.Popen(
        executable_path,
        stdin=subprocess.PIPE,  # 允许写入输入
        stdout=subprocess.PIPE,  # 捕获标准输出
        stderr=subprocess.PIPE,  # 捕获错误输出
        text=True,  # 使用文本模式(自动编码解码)
    )
    
    # 发送输入并获取输出
    stdout_data, stderr_data = process.communicate(
        input=input_data
    )

# 示例用法
for i in range(0, 4):
    exe_path = "./get_matrix_test"    # Linux/macOS示例

    bias_type = 3
    bias_type_name = ["error", "zeroinit", "rowrepeat", "fullbias"]

    # 要传递给程序的输入
    input_text = f"""8
{64 * 2**i} {64 * 2**i} {64 * 2**i}
{bias_type}
0
"""
    print(input_text)

    # 调用程序
    run_executable(
        exe_path, 
        input_text
    )

    os.rename("matmul_value.h", f"matmul_value_mxfp8_mnk_{64 * 2**i}_{64 * 2**i}_{64 * 2**i}_{bias_type_name[bias_type]}.h")