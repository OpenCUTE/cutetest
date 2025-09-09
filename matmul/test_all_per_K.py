import os
from multiprocessing import Pool, current_process, Manager
import subprocess

bus_width = [64, 128, 256]

def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv")
    root_dir = "../../../../" # chipyard路径
    for type in ["magic", "base"]:
        for i in range(3):

            # 创建一个临时的bash脚本来执行所有命令
            script_content = f"""
            source {root_dir}env.sh
            (set -o pipefail &&  {root_dir}sims/verilator/simulator-chipyard.harness-L2DramPerformenceTest1CUTEM{bus_width[i]}Config \
                +permissive \
                +dramsim +dramsim_ini_dir={root_dir}generators/cute/cutetest/dramsim_config/dramsim2/{bus_width[i]}bit_ddr3_{type} +max-cycles=50000000 +loadmem={root_dir}generators/cute/cutetest/matmul/cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv     \
                +verbose \
                +permissive-off \
                {root_dir}generators/cute/cutetest/matmul/cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv \
                \
                </dev/null 2> >(spike-dasm > /dev/null) | tee {root_dir}sims/verilator/output/chipyard.harness.TestHarness.L2DramPerformenceTest1CUTEM{bus_width[i]}Config/cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose_{type}{bus_width[i]}.log)
            """

            script_path = f"{root_dir}tmp/mat512_{x}.sh"
            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            # 给予执行权限
            os.chmod(script_path, 0o755)

            # 执行bash脚本
            subprocess.run(script_path, shell=True, executable="/bin/bash")

    print(f"{process_name}: 完成执行 cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv")

    # 更新进度
    with lock:
        progress.value += 1
        percent_complete = (progress.value / total) * 100
        print(f"总进度: {percent_complete:.2f}%")

if __name__ == "__main__":
    total_tasks = 64  # 总任务数
    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象

    # 创建一个包含64个进程的进程池
    with Pool(64) as p:
        p.starmap(run_command, [(x, progress, total_tasks, lock) for x in range(1, 65)])