"""把 RAG 暴露成 MCP server(Streamable HTTP)。"""
from mcp.server.fastmcp import FastMCP

from . import rag

mcp = FastMCP("vikko-rag", port=9000)


@mcp.tool()
def rag_query(query: str) -> str:
    """基于本地知识库(量子位 AI 文章)检索并回答用户问题。"""
    return rag.answer(query)


if __name__ == "__main__":
    # Streamable HTTP 传输;Java 侧用 McpClient 连 http://127.0.0.1:9000/mcp
    mcp.run(transport="streamable-http")
