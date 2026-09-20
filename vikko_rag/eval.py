"""RAG Triad 评测:Context Relevance / Groundedness / Answer Relevance。

三件套是业界标准(TruEra 提出)的 RAG 质量评测,用 LLM-as-judge 打分,分别查:
- Context Relevance:召回的 chunk 与问题相不相关 → 查「检索质量」;
- Groundedness:回答有没有被召回资料支撑 → 查「幻觉」;
- Answer Relevance:回答有没有答到点上 → 查「生成质量」。

这是「离线评测」:拿测试集批量打分,不参与每次查询的在线链路。用途:
开发期调参对比 / CI 回归 / 上线后抽样监控。每条查询会调多次 DeepSeek,
跑之前确认 .env 里有 key 且能连外网。

用法(在项目根 vikko-rag 下):
    .venv/bin/python -m vikko_rag.eval
"""
import re

from openai import OpenAI

from . import config, rag

# 测试集:换成你自己的问题即可(三件套只需要 query,不需要参考答案)
QUERIES = [
    "唐杰发布的 RSI 是什么?",
    "马斯克为什么睡进工地?",
    "Flash 模型靠什么反打旗舰?",
    "今天天气怎么样?",  # 语料里没有,验证三件套对「无资料」问题的打分
]

CONTEXT_RELEVANCE_PROMPT = """你是 RAG 检索质量评估员。判断「检索到的资料」与「用户问题」的相关程度。

用户问题:{query}

检索到的资料:
{contexts}

只输出一个 0-10 的整数分数:0=资料与问题完全无关,10=资料高度相关、直接命中问题。不要输出任何其它文字。"""

GROUNDEDNESS_PROMPT = """你是 RAG 回答忠实度(Groundedness)评估员。你的唯一任务是判断「模型回答」里的每个论断是否能被「检索到的资料」支撑,即有没有无中生有 / 幻觉。

注意:
- 只判断「忠实度」,不要因为回答篇幅短、不够详细 / 完整而扣分——那些是回答质量的问题,不属于本项;
- 只要回答里的数字、专有名词、结论、因果等具体论断都能在资料里找到依据(或由资料直接得出),就是满分;
- 回答明确说「资料中没有相关内容」且资料确实不足以回答,也视为忠实。

检索到的资料:
{contexts}

模型回答:{answer}

只输出一个 0-10 的整数分数:0=回答几乎全是资料里没有的编造,10=回答的每个论断都忠实于资料。不要输出任何其它文字。"""

ANSWER_RELEVANCE_PROMPT = """你是回答质量评估员。判断「模型回答」是否直接、完整地回答了「用户问题」。

用户问题:{query}

模型回答:{answer}

只输出一个 0-10 的整数分数:0=回答完全没答到点上,10=回答直接、完整地解决了问题。不要输出任何其它文字。"""


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY,请先 export DEEPSEEK_API_KEY=sk-xxx")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def _judge(prompt: str) -> float | None:
    """调 DeepSeek 做 LLM-as-judge,解析 0-10 分数并归一化到 0-1。

    解析失败(输出里没有数字)返回 None,由上层跳过该维度。
    """
    resp = _client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\d+", raw)
    if not m:
        return None
    score = int(m.group(0))
    return max(0.0, min(score, 10)) / 10.0


def _context_text(contexts: list[dict]) -> str:
    """把检索到的 chunk 拼成编号文本,供 judge 读取。"""
    return "\n\n".join(f"[{i + 1}] {c['text'][:300]}" for i, c in enumerate(contexts))


def context_relevance(query: str, contexts: list[dict]) -> float | None:
    """Context Relevance:检索到的资料与问题的相关度。"""
    return _judge(CONTEXT_RELEVANCE_PROMPT.format(query=query, contexts=_context_text(contexts)))


def groundedness(contexts: list[dict], answer: str) -> float | None:
    """Groundedness:回答被检索资料支撑的程度。"""
    return _judge(GROUNDEDNESS_PROMPT.format(contexts=_context_text(contexts), answer=answer))


def answer_relevance(query: str, answer: str) -> float | None:
    """Answer Relevance:回答是否答到点上。"""
    return _judge(ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer))


def evaluate(queries: list[str]) -> list[dict]:
    """对一组 query 跑三件套评测,返回逐条结果。"""
    results = []
    for q in queries:
        contexts = rag.retrieve(q)        # 检索(含改写 / 混合 / 重排)
        answer = rag.answer(q, contexts)  # 复用上面的 context,避免重复检索
        results.append({
            "query": q,
            "context_relevance": context_relevance(q, contexts),
            "groundedness": groundedness(contexts, answer),
            "answer_relevance": answer_relevance(q, answer),
        })
    return results


def _fmt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def main() -> None:
    print("RAG Triad 评测(每条 query 会调多次 DeepSeek,耐心等)\n")
    results = evaluate(QUERIES)

    sums = {"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0}
    counts = {"context_relevance": 0, "groundedness": 0, "answer_relevance": 0}
    for r in results:
        print(f"❓ {r['query']}")
        print(f"   Context Relevance : {_fmt(r['context_relevance'])}")
        print(f"   Groundedness      : {_fmt(r['groundedness'])}")
        print(f"   Answer Relevance  : {_fmt(r['answer_relevance'])}\n")
        for k in sums:
            if r[k] is not None:
                sums[k] += r[k]
                counts[k] += 1

    print("平均:")
    for label, k in [
        ("Context Relevance", "context_relevance"),
        ("Groundedness", "groundedness"),
        ("Answer Relevance", "answer_relevance"),
    ]:
        avg = sums[k] / counts[k] if counts[k] else None
        print(f"   {label:18s} {_fmt(avg)}")


if __name__ == "__main__":
    main()
