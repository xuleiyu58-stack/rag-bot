import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
chunks = [json.loads(line) for line in open("chunks.jsonl", encoding="utf-8")]
texts = [c["text"] for c in chunks]
emb = model.encode(texts, normalize_embeddings=True)


def ask(q, history):
    q_emb = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ q_emb
    top = np.argsort(-scores)[:5]
    picked = [chunks[i] for i in top]
    context = "\n\n".join(f"【资料{c['id']}｜出处：{c['source']}】\n{c['text']}" for c in picked)
    prompt = f"""下面是参考资料，请只根据这些资料回答问题，不要编造资料里没有的内容。
回答完后，在末尾列出你引用了哪几条资料（写编号和出处）。

参考资料：
{context}

问题：{q}"""
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.3},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


gr.ChatInterface(fn=ask, title="手册 RAG 问答机器人").launch()
