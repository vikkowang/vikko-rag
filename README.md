# vikko-rag

RAG 服务:本地 BGE embedding + **milvus-lite(内嵌 Milvus)** 混合检索(BM25 + 向量)+ 重排 + DeepSeek 生成,
通过 **MCP** 暴露给 `vikko-chat`(Java / Spring AI)调用。

## 架构

```
data/*.md ──语义切块──► BGE 向量化 ──► milvus-lite(内嵌 Milvus,免 Docker)
                                        ▲
   rag_query(query)                     │ 混合检索(BM25+向量,RRF)+ 重排
                                        │
                              DeepSeek 生成(带引用)
                                        │
                        FastMCP server (Streamable HTTP)
                                        ▲
                        vikko-chat 的 McpClient(复用钉钉那套)
```

检索流水线(`rag.retrieve`):查询改写 → 混合检索(BM25 + 向量,RRF 融合)→ 重排(DeepSeek listwise)→ top-k。

## 安装

```bash
cd vikko-rag
python3 -m venv .venv
source .venv/bin/activate
# 国内用清华镜像 + 绕过 SSL(系统 Python 缺 CA 证书,见「常见问题」)
pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

> ⚠️ **Intel Mac 兼容性**:本项目跑在 Intel Mac(macOS x86_64),依赖已钉死
> `torch==2.2.2`(最后一版带 x86_64 wheel)+ `numpy<2` + `sentence-transformers<4` + `mcp<2`
> (见 `pyproject.toml`)。换 Apple Silicon 机器需去掉这些 pin、改用新版 torch。

## 配置 DeepSeek Key

```bash
cp .env.example .env   # 然后填入 sk-xxx
```

## BGE 模型

模型已预下载到 `models/bge-small-zh-v1.5/`(通过 curl 从 hf-mirror 拉的,`config.py` 指向本地路径),
运行时无需联网。换机器或想重下见「常见问题」。

## 导入语料

```bash
python -m vikko_rag.ingest
```

把 `data/*.md` 切块、向量化、写入 `milvus.db`。

## 启动 MCP server

```bash
python -m vikko_rag.server
```

默认 Streamable HTTP 端点在 `http://127.0.0.1:9000/mcp`(host/port 可在 `server.py` 里调)。
启动时同时会拉起**定时爬虫**(每天 3:00 抓量子位最新文章)。

## 定时爬虫(增量拉取量子位文章)

`server.py` 启动时内置了 APScheduler 定时任务,每天 **3:00** 自动:

1. 抓量子位首页 → 解析最新 10 篇;
2. 按文章 ID **去重**(已抓过的跳过);
3. 用 DeepSeek **语义判断广告**,过滤软文;
4. 把新文章存 `data/` 并增量入库。

配置在 `config.py`:`CRAWL_COUNT`(每次抓几篇)、`SCHEDULE_HOUR`(每天几点)。

手动触发一轮:

```bash
python -m vikko_rag.crawl
```

> ⚠️ **milvus-lite 是内嵌单进程的**(有文件锁):定时爬虫必须和 server 同进程(所以集成在 `server.py` 里)。
> 单独跑 `crawl.py` / `ingest.py` 前要先停掉 server,否则报 `DataDirLockedError`。

## 目录结构

```
vikko_rag/
├── config.py      # 路径、模型、DeepSeek、切分/检索/重排、爬虫参数
├── embedding.py   # 本地 BGE 向量化
├── store.py       # milvus-lite 内嵌向量库(免 Docker)
├── ingest.py      # 语料语义切块 + 入库
├── hybrid.py      # 混合检索:BM25 + 向量,RRF 融合
├── rerank.py      # 重排:DeepSeek listwise 精排
├── rag.py         # 检索流水线(改写→混合→重排)+ DeepSeek 生成
├── eval.py        # RAG Triad 评测:Context/Groundedness/Answer Relevance(LLM-as-judge)
├── crawl.py       # 定时爬虫:抓量子位 → 去重 → 广告过滤 → 入库
├── ad_filter.py   # 用 DeepSeek 语义判断广告
└── server.py      # FastMCP server + 内置定时爬虫
```

## 评测(RAG Triad)

`eval.py` 提供 RAG 三件套评测(LLM-as-judge,**离线**打分,不参与每次查询的在线链路):

- **Context Relevance**:召回的 chunk 与问题相关度 → 查「检索质量」;
- **Groundedness**:回答是否被资料支撑 → 查「幻觉」;
- **Answer Relevance**:回答是否答到点上 → 查「生成质量」。

```bash
.venv/bin/python -m vikko_rag.eval
```

每条 query 会调多次 DeepSeek,跑前确认 `.env` 有 key 且能连外网。测试集在 `eval.py` 的 `QUERIES` 里,换成自己的问题即可。

## 常见问题

- **pip 报 SSL 证书错误(`unable to get local issuer certificate`)**:python.org 安装的
  Python 没配 CA 证书。临时绕过:加 `--trusted-host pypi.tuna.tsinghua.edu.cn`(或
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`);永久修复:运行
  `/Applications/Python 3.12/Install Certificates.command`。
- **pip 直连 PyPI 慢 / 下 torch 卡在源码编译**:用清华镜像 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`,
  并确保钉死 `torch==2.2.2`(新版 torch 无 Intel Mac wheel,会退到源码编译)。
- **重新下载 BGE 模型**:模型在 `models/bge-small-zh-v1.5/`,是从 hf-mirror 用 curl 拉的
  (`https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main/<文件>`);huggingface_hub 直连在这台机器上不稳,建议继续用 curl。
