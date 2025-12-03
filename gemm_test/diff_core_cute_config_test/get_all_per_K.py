import os
from multiprocessing import Pool, current_process, Manager
import subprocess

def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv")
    root_dir = "../../../../../" # chipyard路径
    for task in ["CUTE2TopsSmallBoomConfig","CUTE2TopsSmallRocketConfig"]:
        # 创建一个临时的bash脚本来执行所有命令
        script_content = f"""
        source {root_dir}env.sh
        (set -o pipefail &&  ./build/simulator-chipyard.harness-{task}\
            +permissive \
            +dramsim +dramsim_ini_dir=../../dramsim_config/dramsim2_ini_32GB_per_s +max-cycles=200000000 +loadmem=../cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv     \
            +verbose \
            +permissive-off \
            ../cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv \
            \
            </dev/null 2> >(spike-dasm > /dev/null) | tee ./log/cute_Matmul_mnk_512_512_{x*256}_{task}.log)
        """
        script_path = f"./tmp/{task}_mat_512_512_{x*256}.sh"
        with open(script_path, "w") as script_file:
            script_file.write(script_content)

        # 给予执行权限
        os.chmod(script_path, 0o755)

    # 更新进度
    with lock:
        progress.value += 1
        percent_complete = (progress.value / total) * 100
        print(f"总进度: {percent_complete:.2f}%")

if __name__ == "__main__":
    total_tasks = 32  # 总任务数
    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象

    # 创建一个包含64个进程的进程池
    with Pool(32) as p:
        p.starmap(run_command, [(x, progress, total_tasks, lock) for x in range(1, total_tasks+1)])