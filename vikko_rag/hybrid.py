"""混合检索:BM25(关键词召回)+ 稠密向量(语义召回),RRF 融合。

为什么混合:
- 纯向量检索对「关键词精确匹配」不敏感(产品名、型号、专有名词容易漏);
- 纯 BM25 对「同义改写 / 语义相近」不敏感(换种说法就匹配不上);
- 两路召回互补,能同时覆盖「字面命中」和「语义相近」。

为什么用 RRF(Reciprocal Rank Fusion)而不是直接加权:
- 向量路返回的是「距离」、BM25 路返回的是「词频分数」,量纲完全不同,
  直接加权会被数值大的那一路碾压;RRF 只关心「排名」,天然量纲无关:
      score(doc) = Σ 1 / (RRF_K + rank)     # rank 从 1 起,RRF_K 是平滑常数

注意:BM25 需要全量语料建倒排。这里在首次 search 时从 milvus 把全部 chunk
文本拉到内存建一次索引(语料很小,几百块,代价可忽略),之后复用;re-ingest
后调用 reset() 使缓存失效。
"""
from rank_bm25 import BM25Okapi

import jieba

from . import config, embedding, store

# 模块级缓存:BM25 索引 + 语料(首次 search 时惰性构建)
_bm25: BM25Okapi | None = None
_chunks: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """jieba 中文分词,去掉空白 token。"""
    return [w for w in jieba.cut(text) if w.strip()]


def _ensure_index() -> None:
    """惰性构建 BM25 索引:从 milvus 拉全部 chunk 文本建倒排。"""
    global _bm25, _chunks
    if _bm25 is not None:
        return
    _chunks = store.get_all_chunks()
    if _chunks:
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """BM25 单路召回,返回 top_k 个 chunk(含来源)。"""
    _ensure_index()
    if not _bm25:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [_chunks[i] for i in top_indices]


def search(query: str, top_k: int = config.HYBRID_CANDIDATES) -> list[dict]:
    """混合检索:向量 + BM25 两路召回,RRF 融合后返回 top_k 个候选。

    返回 [{text, source, score}, ...],score 为 RRF 融合分(越大越相关)。
    """
    _ensure_index()

    # 1. 向量路:稠密语义召回
    qv = embedding.embed([query])[0]
    vector_hits = store.search(qv, config.VECTOR_TOP_K)

    # 2. BM25 路:关键词召回
    bm25_hits = _bm25_search(query, config.BM25_TOP_K)

    # 3. RRF 融合:同一 chunk 在两路都命中时,融合分会累加
    rrf: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for rank, hit in enumerate(vector_hits):
        key = hit["text"]
        meta[key] = {"text": hit["text"], "source": hit["source"]}
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (config.RRF_K + rank + 1)
    for rank, hit in enumerate(bm25_hits):
        key = hit["text"]
        meta[key] = {"text": hit["text"], "source": hit["source"]}
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (config.RRF_K + rank + 1)

    ranked = sorted(rrf.items(), key=lambda kv: -kv[1])
    return [
        {"text": meta[k]["text"], "source": meta[k]["source"], "score": s}
        for k, s in ranked[:top_k]
    ]


def reset() -> None:
    """清空 BM25 缓存,下次 search 时重新从 milvus 拉全量文本建索引。

    re-ingest(新增 / 更新语料)后调用,避免 BM25 索引滞后于向量库。
    """
    global _bm25, _chunks
    _bm25 = None
    _chunks = []
