import os
import subprocess

simulator = "simulator-chipyard-CUTETestConfig"
# test_file_dir = "/root/chipyard/generators/boom/src/main/resources/cutetest"
test_file_dir = "/root/chipyard/generators/boom/src/main/resources/cutetest"
# test_name = "cute_Matmul_mnk_113_128_128_fullbias"
test_name = "softmax"
# vcd_arg = f"+verbose -v/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig/{test_name}.vcd"
vcd_arg = ""
# 创建一个临时的bash脚本来执行所有命令
script_content = f"""
source /home/yuanbin/chipyard/env.sh
(set -o pipefail &&  /home/yuanbin/chipyard/sims/verilator/simulator-chipyard.harness-CUTETestConfig128bitdram512bitL2Widen3issueBoom-debug \
        +permissive \
        +dramsim +dramsim_ini_dir=/home/yuanbin/chipyard/generators/testchipip/src/main/resources/dramsim2_ini +max-cycles=5000000 +loadmem=/home/yuanbin/chipyard/generators/cute/cutetest/transformer/cutetest/softmax_cute.riscv     \
        +verbose \
        +vcdfile=/home/yuanbin/chipyard/sims/verilator/output/chipyard.harness.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/softmax_cute.fst \
        +permissive-off \
        /home/yuanbin/chipyard/generators/cute/cutetest/transformer/cutetest/softmax_cute.riscv \
         \
        </dev/null 2> >(spike-dasm > /home/yuanbin/chipyard/sims/verilator/output/chipyard.harness.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/softmax_cute.out) | tee /home/yuanbin/chipyard/sims/verilator/output/chipyard.harness.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/softmax_cute.log)
"""

script_path = f"/tmp/run_{test_name}.sh"
with open(script_path, "w") as script_file:
    script_file.write(script_content)

# 给予执行权限
os.chmod(script_path, 0o755)

# 执行bash脚本
subprocess.run(script_path, shell=True, executable="/bin/bash")

print(f"完成执行 {test_name}.riscv")