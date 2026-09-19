"""广告过滤:用 DeepSeek 语义判断一篇文章是不是广告/推广/软文。"""
from openai import OpenAI

from . import config

_client: OpenAI | None = None

AD_PROMPT = """你是内容审核员。判断下面这篇文章是否属于「广告、推广、软文」
(例如带货、优惠活动、品牌宣传、产品推介等)。

标题:{title}
正文(节选):{content}

只回答一个字:「是」或「否」。"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY,请先 export DEEPSEEK_API_KEY=sk-xxx")
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


def is_ad(title: str, content: str) -> bool:
    """判断是否为广告。返回 True = 广告,应过滤。"""
    prompt = AD_PROMPT.format(title=title, content=content[:300])
    resp = _get_client().chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return "是" in answer
