import os
import re
import csv

log_dir = "./log"

# 文件名匹配模式：匹配 ibert_1_48GB_task.log / ibert_1_48GB-nofuse_task.log / ibert_1_48GB-notcm_task.log
pattern_filename = re.compile(
    r"ibert_(\d+)_(\d+)GB(?:-(nofuse|notcm))?_task\.log"
)
pattern_cycles = re.compile(r"bert-base cycles:\s*(\d+)")

output_csv = "bert-base.csv"
rows = []

for root, dirs, files in os.walk(log_dir):
    for filename in files:
        match = pattern_filename.match(filename)
        if not match:
            continue

        layer, GB, mode = match.groups()
        mode = mode or "base"  # 没有后缀的默认为 base（普通版本）

        filepath = os.path.join(root, filename)
        try:
            with open(filepath, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 无法读取 {filepath}: {e}")
            continue

        conv_cycles = pattern_cycles.search(content)
        if not conv_cycles:
            print(f"⚠️ 未找到 bert-base cycles: {filename}")
            continue

        cycles = conv_cycles.group(1)
        rows.append([layer, GB, mode, cycles])

# 写出 CSV 文件
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["layer", "GB", "Mode", "Cycles"])
    writer.writerows(rows)

print(f"✅ 已生成 {output_csv}，共 {len(rows)} 条记录。")
