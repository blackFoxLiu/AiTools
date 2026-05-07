import json


input_file = "./output/rag_blocks.jsonl"  # 替换为你的 jsonl 文件路径
output_file = "./output/rag_blocks.txt"  # 输出的 txt 文件路径

with open(input_file, 'r', encoding='utf-8') as inf, \
        open(output_file, 'w', encoding='utf-8') as outf:
    for line in inf:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        text = data.get("text", "")
        if text:
            outf.write(text + "\n\n")  # 每个段落之间空一行（可根据需要调整）

print(f"已将所有 text 存储到 {output_file}")