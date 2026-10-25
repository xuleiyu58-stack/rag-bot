import json
import re

INPUT = "大模型面试蒸馏手册.md"
OUTPUT = "chunks.jsonl"


def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        text = f.read()

    # 按二级、三级标题切分，标题留作这张卡片的出处
    parts = re.split(r"\n(?=#{2,3} )", text)
    chunks = []
    current_source = "手册开头"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split("\n", 1)[0]
        if first_line.startswith("#"):
            current_source = first_line.lstrip("#").strip()
        # 每块约 400 字，块之间重叠 50 字
        step = 400
        overlap = 50
        start = 0
        while start < len(part):
            piece = part[start:start + step]
            chunks.append({"id": len(chunks) + 1,
                           "source": current_source,
                           "text": piece})
            if start + step >= len(part):
                break
            start += step - overlap

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"共切出 {len(chunks)} 张卡片，已保存到 {OUTPUT}")
    for c in chunks[:3]:
        print("----", c["id"], c["source"], "----")
        print(c["text"][:120])


if __name__ == "__main__":
    main()


