"""把 data/*.md 切块 → 向量化 → 写入 milvus-lite。"""
from . import config, embedding, store


def chunk_text(text: str) -> list[str]:
    """按固定字符数 + 重叠切分。"""
    text = text.strip()
    if len(text) <= config.CHUNK_SIZE:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + config.CHUNK_SIZE])
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


def ingest(files=None) -> int:
    if files is None:
        files = list(config.DATA_DIR.glob("*.md"))
    if not files:
        print(f"⚠️  语料目录为空,请把 markdown 放进 {config.DATA_DIR}")
        return 0

    total = 0
    for f in files:
        chunks = chunk_text(f.read_text(encoding="utf-8"))
        if not chunks:
            continue
        store.upsert(
            texts=chunks,
            sources=[f.name] * len(chunks),
            embeddings=embedding.embed(chunks),
        )
        print(f"✓ {f.name}: {len(chunks)} 块")
        total += len(chunks)
    print(f"完成,共 {total} 块已入库(总计 {store.count()} 块)")
    return total


if __name__ == "__main__":
    ingest()
