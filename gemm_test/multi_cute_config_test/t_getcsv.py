import os
import re
import csv

log_dir = "./t_log"
# 文件名匹配模式：提取 M N K GB Tops，可以匹配05Tops这类格式
pattern_filename = re.compile(
    r"cute_Matmul_mnk_(\d+)_(\d+)_(\d+)_zeroinit_transpose_(\d+)GB_([0-9]+)Tops\.log"
)
pattern_cycles = re.compile(r"matmul cycles:\s*(\d+)")

output_csv = "matmul_cycles.csv"
rows = []

for root, dirs, files in os.walk(log_dir):
    for filename in files:
        match = pattern_filename.match(filename)
        if not match:
            continue

        M, N, K, GB, Tops = match.groups()

        # 处理 Tops，例如 05 -> 0.5
        if Tops == "05":
            Tops = "0.5"
        else:
            Tops = str(int(Tops))  # 去掉多余的前导零

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
        rows.append([M, N, K, GB, Tops, cycles])

# 写出 CSV 文件
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["M", "N", "K", "GB", "Tops", "Cycles"])
    writer.writerows(rows)

print(f"✅ 已生成 {output_csv}，共 {len(rows)} 条记录。")
