"""全局配置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"          # 语料目录
MILVUS_DB = BASE_DIR / "milvus.db"    # milvus-lite 内嵌数据库文件

# embedding 模型:已下载到本地 models/ 目录,避免运行时连 HuggingFace
EMBEDDING_MODEL = str(BASE_DIR / "models" / "bge-small-zh-v1.5")
EMBEDDING_DIM = 512                   # bge-small-zh-v1.5 的向量维度

# DeepSeek(OpenAI 兼容接口)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 切分与检索参数
CHUNK_SIZE = 400       # 每块字符数
CHUNK_OVERLAP = 50     # 相邻块重叠字符数
TOP_K = 3              # 检索返回的块数
COLLECTION_NAME = "rag_docs"

# 爬虫配置(定时拉取量子位文章)
QBITAI_HOME = "https://www.qbitai.com/"
CRAWL_COUNT = 10       # 每次抓最新多少篇
SCHEDULE_HOUR = 3      # 每天几点抓(24 小时制)
