import os
from multiprocessing import Pool, current_process, Manager
import subprocess

bus_width = [8 ,16, 32, 64]
tops_8GB    = [1,2,4,8]
tops_16GB   = [2,4,8,16]
tops_32GB   = [4,8,16,32]
tops_64GB   = [8,16,32,64]

def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv")
    root_dir = "../../../../../" # chipyard路径
    for task in [["8GB",tops_8GB], ["16GB",tops_16GB], ["32GB",tops_32GB], ["64GB",tops_64GB]]:
        type = task[0]
        for i in range(4):
            tops = task[1]
            # 创建一个临时的bash脚本来执行所有命令
            script_content = f"""
            source {root_dir}env.sh
            (set -o pipefail &&  ./build/simulator-chipyard.harness-CUTE{tops[i]}TopsConfig\
                +permissive \
                +dramsim +dramsim_ini_dir=../../dramsim_config/dramsim2_ini_{type}_per_s +max-cycles=50000000 +loadmem=../cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv     \
                +verbose \
                +permissive-off \
                ../cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose.riscv \
                \
                </dev/null 2> >(spike-dasm > /dev/null) | tee ./log/cute_Matmul_mnk_512_512_{x*256}_zeroinit_transpose_{type}_{tops[i]}Tops.log)
            """

            script_path = f"./tmp/{tops[i]}Tops_{type}ddr_mat512_512_{x*256}.sh"
            with open(script_path, "w") as script_file:
                script_file.write(script_content)

            # 给予执行权限
            os.chmod(script_path, 0o755)

            # 执行bash脚本
            # subprocess.run(script_path, shell=True, executable="/bin/bash")
            
            print(f"{process_name}: 完成执行 {tops[i]}Tops_{type}ddr_cute_Matmul_mnk_512_512_{x*256}")

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