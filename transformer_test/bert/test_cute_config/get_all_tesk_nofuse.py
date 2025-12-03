import os
from multiprocessing import Pool, current_process, Manager
import subprocess

# vec_ops_conv_2.riscv

DDR_type = ["24GB","48GB"]

def run_command(x, progress, total, lock):
    process_name = current_process().name
    
    print(f"{process_name}: 开始生成 conv_{x}_task.sh")
    
    # 创建一个临时的bash脚本来执行所有命令
    for ddrtype in DDR_type:
        script_content = f"""
        (set -o pipefail &&  ./build/simulator-chipyard.harness-CUTE4TopsShuttle512D512V512M512Sysbus512Membus1CoreConfig\
            +permissive \
            +dramsim +dramsim_ini_dir=../../dramsim_config/dramsim2_ini_{ddrtype}_per_s +max-cycles=800000000 +loadmem=../ibert-base-{x}-nofuse.riscv     \
            +verbose \
            +permissive-off \
            ../ibert-base-{x}-nofuse.riscv \
            \
            </dev/null  | tee ./log/ibert_{x}_{ddrtype}-nofuse_task.log)
        """

        script_path = f"./test/ibert_{x}_{ddrtype}-nofuse_task.sh"
        with open(script_path, "w") as script_file:
            script_file.write(script_content)

        # 给予执行权限
        os.chmod(script_path, 0o755)


if __name__ == "__main__":
    conv_total_task = range(1,6+1)

    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象

    # 创建一个包含64个进程的进程池
    with Pool(64) as p:
        p.starmap(run_command, [(x, progress, conv_total_task.count, lock) for x in conv_total_task])