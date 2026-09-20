"""把 data/*.md 切块 → 向量化 → 写入 milvus-lite。"""
from . import config, embedding, hybrid, store


def chunk_text(text: str) -> list[str]:
    """语义切分:按 markdown 结构(标题/段落)切,块大小不超过 CHUNK_SIZE。

    相比固定字符数切分:
    1. 不把一句话从中间劈开(以空行为界拆成「段落单元」);
    2. 标题(#/##/###)单独开块,不跨标题合并,尊重文档结构;
    3. 只有单个超长段落才退化成硬切(带重叠)。
    """
    text = text.strip()
    if not text:
        return []

    # 1. 以空行为界拆成段落/标题单元
    units = [u.strip() for u in text.split("\n\n") if u.strip()]

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        # 标题行开新块,不和上一段混在一起
        if unit.startswith(("# ", "## ", "### ")):
            if current:
                chunks.append("\n\n".join(current))
                current = []
            current.append(unit)
            continue

        # 当前块加上这个单元会超限 → 先落块
        if current and sum(len(x) for x in current) + len(unit) > config.CHUNK_SIZE:
            chunks.append("\n\n".join(current))
            current = []

        # 单个单元本身超长 → 硬切(带重叠)
        if len(unit) > config.CHUNK_SIZE:
            if current:
                chunks.append("\n\n".join(current))
                current = []
            chunks.extend(_hard_split(unit))
            continue

        current.append(unit)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _hard_split(text: str) -> list[str]:
    """单个超长段落的兜底硬切(带重叠),语义切分正常走不到这里。"""
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
    hybrid.reset()  # 语料变了,清空 BM25 缓存,下次检索时重新建索引
    return total


if __name__ == "__main__":
    ingest()
