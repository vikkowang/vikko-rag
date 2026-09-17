"""向量库:milvus-lite 内嵌(免 Docker)。"""
from pymilvus import DataType, MilvusClient

from . import config

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(str(config.MILVUS_DB))
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(config.COLLECTION_NAME):
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=256)
    client.create_collection(collection_name=config.COLLECTION_NAME, schema=schema)


def upsert(texts: list[str], sources: list[str], embeddings: list[list[float]]) -> None:
    data = [
        {"vector": emb, "text": text, "source": src}
        for emb, text, src in zip(embeddings, texts, sources)
    ]
    get_client().insert(collection_name=config.COLLECTION_NAME, data=data)


def search(query_embedding: list[float], top_k: int = config.TOP_K) -> list[dict]:
    res = get_client().search(
        collection_name=config.COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source"],
    )
    return [
        {"text": h["entity"]["text"], "source": h["entity"]["source"], "distance": h["distance"]}
        for h in res[0]
    ]


def count() -> int:
    return get_client().get_collection_stats(config.COLLECTION_NAME)["row_count"]
