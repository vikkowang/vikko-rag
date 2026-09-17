"""本地 BGE embedding。"""
from sentence_transformers import SentenceTransformer

from . import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # 国内网络慢的话,先 export HF_ENDPOINT=https://hf-mirror.com
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本向量化,返回归一化后的向量列表。"""
    return get_model().encode(texts, normalize_embeddings=True).tolist()
