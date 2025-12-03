import os
import re
import csv

log_dir = "./log"

# 文件名匹配模式，支持 base / -nofuse / -notcm 三种
pattern_filename = re.compile(
    r"conv_(\d+)_(\d+)GB(?:_(nofuse|notcm))?_task\.log"
)
# 匹配类似 “conv_3 cycles: 123456”
pattern_cycles = re.compile(r"conv_(\d+)\s+cycles:\s*(\d+)")

output_csv = "matmul_cycles.csv"
rows = []

for root, dirs, files in os.walk(log_dir):
    for filename in files:
        match = pattern_filename.match(filename)
        if not match:
            continue

        layer, GB, mode = match.groups()
        mode = mode or "base"  # 没有后缀时设为 base

        filepath = os.path.join(root, filename)
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 无法读取 {filepath}: {e}")
            continue

        conv_cycles = pattern_cycles.search(content)
        if not conv_cycles:
            print(f"⚠️ 未找到 conv cycles: {filename}")
            continue

        cycles = conv_cycles.group(2)
        rows.append([layer, GB, mode, cycles])

# 写出 CSV 文件
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Layer", "GB", "Mode", "Cycles"])
    writer.writerows(rows)

print(f"✅ 已生成 {output_csv}，共 {len(rows)} 条记录。")
