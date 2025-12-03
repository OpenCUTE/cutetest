import os
from multiprocessing import Pool, current_process, Manager
import subprocess


def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 {x}")
    script_path = f"./tmp/{x}"
    # 执行bash脚本
    subprocess.run(script_path, shell=True, executable="/bin/bash")
    print(f"{process_name}: 完成执行 {x}")

    # 更新进度
    with lock:
        progress.value += 1
        percent_complete = (progress.value / total) * 100
        print(f"总进度: {percent_complete:.2f}%")

if __name__ == "__main__":
    #获取./tmp目录下的所有脚本，脚本数量就是总任务数
    tmp_task = os.listdir("./tmp")
    total_tasks = len([name for name in tmp_task if name.endswith(".sh")])  # 总任务数
    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象
    

    # 创建一个包含64个进程的进程池
    with Pool(20) as p:
        p.starmap(run_command, [(x, progress, total_tasks, lock) for x in tmp_task if x.endswith(".sh")])