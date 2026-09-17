"""手动测试 RAG:rag.retrieve(检索)与 rag.answer(检索 + 生成)。

用法(在项目根 vikko-rag 目录下):
    .venv/bin/python test_rag.py
"""
from vikko_rag import rag

QUERIES = [
    "唐杰发布的 RSI 是什么？",
    "马斯克为什么睡进工地？",
    "Flash 模型靠什么反打旗舰？",
    "今天天气怎么样？",   # 语料里没有的内容,验证「资料中没有相关内容」
]


def show_retrieve(query: str) -> None:
    print(f"\n{'=' * 60}\n🔍 检索: {query}\n{'=' * 60}")
    for i, r in enumerate(rag.retrieve(query), 1):
        text = r["text"].replace("\n", " ")[:60]
        print(f"  [{i}] 相似度 {r['distance']:.3f} | {r['source']} | {text}...")


def show_answer(query: str) -> None:
    print(f"\n💬 回答: {query}")
    print(rag.answer(query))


if __name__ == "__main__":
    for q in QUERIES:
        show_retrieve(q)
        show_answer(q)
