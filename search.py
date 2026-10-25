import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# BGE-small-zh：中文小模型，本地就能跑
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

chunks = [json.loads(line) for line in open("chunks.jsonl", encoding="utf-8")]
texts = [c["text"] for c in chunks]
# 全部卡片变向量，normalize 之后点乘就是余弦相似度
emb = model.encode(texts, normalize_embeddings=True)

q = input("请输入你的问题：")
q_emb = model.encode([q], normalize_embeddings=True)[0]
scores = emb @ q_emb
top = np.argsort(-scores)[:5]
for i in top:
    c = chunks[i]
    print("=" * 40)
    print(f"[{c['id']}] 出处：{c['source']} 相似度：{scores[i]:.3f}")
    print(c["text"][:200])
