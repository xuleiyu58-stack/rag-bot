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
if os.path.exists("emb.npy"):
    emb = np.load("emb.npy")
else:
    emb = model.encode(texts, normalize_embeddings=True)
    np.save("emb.npy", emb)



def ask(q, history):
    q_emb = model.encode([q], normalize_embeddings=True)[0]
    scores = emb @ q_emb
    top = np.argsort(-scores)[:5]
    picked = [chunks[i] for i in top]
    context = "\n\n".join(f"【资料{c['id']}｜出处：{c['source']}】\n{c['text']}" for c in picked)
    hist_text = ""
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        hist_text += f"{role}：{msg['content']}\n"

    prompt = f"""下面是参考资料，请只根据这些资料回答问题，不要编造资料里没有的内容。
回答完后，在末尾列出你引用了哪几条资料（写编号和出处）。
之前的对话记录（供参考，回答新问题时可以接上文）：
{hist_text}


参考资料：
{context}

问题：{q}"""
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.3,
              "stream": True},
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()
    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        text += delta
        yield text


gr.ChatInterface(fn=ask, title="手册 RAG 问答机器人").launch()
