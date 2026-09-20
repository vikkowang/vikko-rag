"""重排(rerank):对混合检索召回的候选做二阶段精排。

为什么需要重排:
- 混合检索(BM25 + 向量 + RRF)属于「召回」阶段,目标是「别漏」,返回的候选
  排序还比较粗——RRF 只按排名做数学融合,并没有真正理解 query 和候选之间
  的语义关系,排在前面的未必最相关。
- 重排属于「精排」阶段,目标是「排序准」:让模型直接读 query + 候选,判断
  哪个更相关,把最相关的排到最前面,这样生成阶段只取前 top-k 就能拿到最
  有用的上下文。

为什么用「listwise 重排」而不是「pairwise 逐对打分」:
- listwise:一次调用,把全部候选编号后喂给模型,让它直接输出一个排序
  (如「3,1,5,2,4」)。✅ 只花一次 LLM 调用,且排序是全局一致的。
- pairwise:对每个候选单独问「这条相关吗」,要 N 次调用;而且每个候选是
  独立打的分,彼此不可比,拼出来的排序不可靠。❌

代价:每次检索会多一次 DeepSeek 调用(重排是纯 LLM 判断,没有额外模型下载)。
"""
from openai import OpenAI

from . import config


def _client() -> OpenAI:
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY,请先 export DEEPSEEK_API_KEY=sk-xxx")
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


# listwise 重排提示词:让模型按相关性输出候选编号序列
RERANK_PROMPT = """你是检索结果重排器。根据用户问题,把下面的候选资料按「对回答该问题的有用程度」从高到低排序。

用户问题:{query}

候选资料:
{candidates}

只输出排序结果:按相关性从高到低输出候选编号,用逗号分隔,例如「3,1,5,2,4」。不要输出任何解释或其它文字。"""


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """对候选做 listwise 重排,返回按相关性降序排列的候选列表。

    参数:
        query: 用户问题(或改写后的检索 query)
        candidates: 混合检索召回的候选,形如 [{text, source, score}, ...]

    返回:
        重新排序后的 candidates(不新增、不删除,只换顺序)。
        输入不足 2 条时原样返回——1 条或 0 条没有排序意义,直接跳过省一次调用。
    """
    if len(candidates) <= 1:
        return candidates

    # 1. 把候选逐个编号后喂给模型;每个候选只截前 200 字
    #    (足够判断相关性,又省 token,避免超长候选撑爆上下文)
    numbered = "\n\n".join(f"[{i + 1}] {c['text'][:200]}" for i, c in enumerate(candidates))
    prompt = RERANK_PROMPT.format(query=query, candidates=numbered)

    # 2. 调 DeepSeek 做重排。temperature=0:排序不需要创造性,0 让输出更稳定可复现
    resp = _client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()

    # 3. 解析模型输出的编号序列,按它重排候选
    order = _parse_order(raw, len(candidates))
    return [candidates[i] for i in order]


def _parse_order(raw: str, n: int) -> list[int]:
    """把模型的编号输出解析成 0-based 索引序列;解析失败兜底为原顺序。

    模型输出不一定规整(可能带空格、中文逗号、额外说明、漏编号),这里尽量鲁棒:
    1. 用正则把输出里所有整数抠出来(不管分隔符是逗号还是别的);
    2. 按出现顺序去重,丢弃越界编号(模型可能编错号);
    3. 把模型没提到的编号补到末尾(保持原有相对顺序),保证一个候选都不丢。

    这样即使模型输出半残,重排也不会导致「丢候选」,最坏就是退化回原顺序。
    """
    import re

    seen: set[int] = set()
    order: list[int] = []
    for num in re.findall(r"\d+", raw):
        idx = int(num) - 1  # 模型输出是 1-based,转成 0-based 索引
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    # 补上模型漏掉的候选,兜底「输出编号不全」的情况
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order
