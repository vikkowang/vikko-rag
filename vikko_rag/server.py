"""把 RAG 暴露成 MCP server(Streamable HTTP),并内置定时爬虫。"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from mcp.server.fastmcp import FastMCP

from . import config, crawl, rag

mcp = FastMCP("vikko-rag", port=9000)


@mcp.tool()
def rag_query(query: str) -> str:
    """基于本地知识库(量子位 AI 文章)检索并回答用户问题。"""
    return rag.answer(query)


def _start_scheduler() -> BackgroundScheduler:
    """启动后台定时爬虫。

    注意:milvus-lite 是内嵌、单进程的(有文件锁),爬虫必须和 server 跑在
    同一个进程里、共用同一个 MilvusClient,否则会报 DataDirLockedError。
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(crawl.crawl, CronTrigger(hour=config.SCHEDULE_HOUR, minute=0))
    scheduler.start()
    print(f"定时爬虫已启动:每天 {config.SCHEDULE_HOUR}:00 抓取量子位最新文章")
    return scheduler


if __name__ == "__main__":
    _start_scheduler()
    # Streamable HTTP 传输;Java 侧用 McpClient 连 http://127.0.0.1:9000/mcp
    mcp.run(transport="streamable-http")
