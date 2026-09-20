"""检索 + DeepSeek 生成。"""
from openai import OpenAI

from . import config, hybrid, rerank


REWRITE_PROMPT = """把下面的用户问题改写成更适合检索的简洁 query:
- 去掉寒暄、口语和指代(如「帮我查一下」「这个」);
- 保留核心语义和关键实体/术语;
- 只输出改写后的 query,不要任何解释。

用户问题:{query}"""


def rewrite_query(query: str) -> str:
    """查询改写:把口语化问题改写成检索友好 query,提升召回。

    改写失败(没 key / 网络异常 / 模型输出为空)时退化为原 query,不阻断检索。
    """
    if not config.DEEPSEEK_API_KEY:
        return query
    try:
        resp = _client().chat.completions.create(
            model=config.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            temperature=0.1,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        return query


def retrieve(query: str) -> list[dict]:
    """完整检索流水线:查询改写 → 混合检索(BM25+向量) → 重排 → top-k。

    返回 [{text, source, score}, ...],按相关性降序。
    """
    q = rewrite_query(query)
    candidates = hybrid.search(q, config.HYBRID_CANDIDATES)
    ranked = rerank.rerank(q, candidates)
    return ranked[: config.TOP_K]


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY,请先 export DEEPSEEK_API_KEY=sk-xxx")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def answer(query: str, contexts: list[dict] | None = None) -> str:
    # contexts 可复用已检索结果(评测时避免重复检索);不传则内部走完整检索流水线
    if contexts is None:
        contexts = retrieve(query)
    if not contexts:
        return "知识库为空,请先运行 python -m vikko_rag.ingest"

    context_text = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
    prompt = (
        "你是严谨的问答助手,请【严格只依据】下面的「资料」回答,不要使用资料之外的任何知识。\n\n"
        "要求:\n"
        "1. 每个论断都要能在资料里找到依据;禁止编造资料中没有的数字、事实、结论或评价;\n"
        "2. 资料不足以回答时,直接回复「资料中没有相关内容」,不要勉强作答或推测;\n"
        "3. 转述资料时保持原意,不要夸大、不要添加原文没有的信息。\n\n"
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
