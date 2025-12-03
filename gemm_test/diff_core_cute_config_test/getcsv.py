import os
import re
import csv

log_dir = "./log"
# 文件名匹配模式：提取 M N K GB Tops，可以匹配05Tops这类格式
boom_pattern_filename = re.compile(
    r"cute_Matmul_mnk_(\d+)_(\d+)_(\d+)_CUTE2TopsSmallBoomConfig\.log"
)
rocket_pattern_filename = re.compile(
    r"cute_Matmul_mnk_(\d+)_(\d+)_(\d+)_CUTE2TopsSmallRocketConfig\.log"
)
pattern_cycles = re.compile(r"matmul cycles:\s*(\d+)")

output_csv = "matmul_cycles.csv"
rows = []

for root, dirs, files in os.walk(log_dir):
    for filename in files:
        match = boom_pattern_filename.match(filename)
        if not match:
            continue

        M, N, K = match.groups()

        filepath = os.path.join(root, filename)
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 无法读取 {filepath}: {e}")
            continue

        match_cycles = pattern_cycles.search(content)
        if not match_cycles:
            print(f"⚠️ 未找到 matmul cycles: {filename}")
            continue

        cycles = match_cycles.group(1)
        rows.append([M, N, K, "Boom", cycles])
        
    for filename in files:
        match = rocket_pattern_filename.match(filename)
        if not match:
            continue

        M, N, K = match.groups()

        filepath = os.path.join(root, filename)
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 无法读取 {filepath}: {e}")
            continue

        match_cycles = pattern_cycles.search(content)
        if not match_cycles:
            print(f"⚠️ 未找到 matmul cycles: {filename}")
            continue

        cycles = match_cycles.group(1)
        rows.append([M, N, K, "Rocket", cycles])

# 写出 CSV 文件
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["M", "N", "K", "Core", "Cycles"])
    writer.writerows(rows)

print(f"✅ 已生成 {output_csv}，共 {len(rows)} 条记录。")
