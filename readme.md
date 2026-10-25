# 手册 RAG 问答机器人

用自己的学习手册做语料的 RAG 问答机器人：先翻书，再回答，并标出出处。

## 它是怎么工作的
1. chunk.py：把手册按标题切块，每块约 400 字、重叠 50 字，存成 chunks.jsonl（147 张卡片）
2. search.py：用 BGE-small-zh-v1.5 把卡片变成本地向量，提问也变成向量，比相似度取前 5 名
3. answer.py：把前 5 名卡片拼进 Prompt，调用 DeepSeek 生成回答，并列出引用出处

## 怎么跑
pip install sentence-transformers
设环境变量 DEEPSEEK_API_KEY 为你的 DeepSeek key
python answer.py

## 演示
问：RAG 是什么？
答：离线索引（加载→切分→建向量索引）+ 在线五步（输入处理→检索→后处理→拼 Prompt 生成→校验），引用资料 27、122、29。

