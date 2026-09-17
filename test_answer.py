"""专门测试 rag.answer()(检索 + DeepSeek 生成)。

用法(在项目根 vikko-rag 目录下):
    .venv/bin/python test_answer.py                  # 用内置的几个问题
    .venv/bin/python test_answer.py "你的问题"        # 指定问题(可传多个,空格分隔)
"""
import sys

from vikko_rag import rag

DEFAULT_QUERIES = [
    "唐杰发布的 RSI 是什么？",
    "马斯克为什么睡进工地？",
    "春江花月夜为什么这么出名", # 手动添加
]


def answer(query: str) -> None:
    print(f"\n{'=' * 60}\n❓ {query}\n{'=' * 60}")
    print(rag.answer(query))


if __name__ == "__main__":
    queries = sys.argv[1:] or DEFAULT_QUERIES
    for q in queries:
        answer(q)
