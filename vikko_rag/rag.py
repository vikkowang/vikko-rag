"""检索 + DeepSeek 生成。"""
from openai import OpenAI

from . import config, embedding, store


def retrieve(query: str) -> list[dict]:
    """检索 top-k 相关块,返回 [{text, source, distance}, ...]。"""
    qv = embedding.embed([query])[0]
    return store.search(qv, config.TOP_K)


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY,请先 export DEEPSEEK_API_KEY=sk-xxx")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def answer(query: str) -> str:
    contexts = retrieve(query)
    if not contexts:
        return "知识库为空,请先运行 python -m vikko_rag.ingest"

    context_text = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
    prompt = (
        "请仅根据下面的资料回答用户问题;若资料不足以回答,明确说“资料中没有相关内容”。\n\n"
        f"资料:\n{context_text}\n\n"
        f"用户问题:{query}\n\n回答:"
    )
    resp = _client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer_text = resp.choices[0].message.content or ""

    sources = sorted({c["source"] for c in contexts})
    return f"{answer_text}\n\n📚 参考:{'、'.join(sources)}"
