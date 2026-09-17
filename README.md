# vikko-rag

RAG 服务:本地 BGE embedding + **milvus-lite(内嵌 Milvus)** 向量检索 + DeepSeek 生成,
通过 **MCP** 暴露给 `vikko-chat`(Java / Spring AI)调用。

## 架构

```
data/*.md ──切块──► BGE 向量化 ──► milvus-lite(内嵌 Milvus,免 Docker)
                                        ▲
                      rag_query(query)  │ 相似度检索 top-k
                                        │
                              DeepSeek 生成(带引用)
                                        │
                        FastMCP server (Streamable HTTP)
                                        ▲
                        vikko-chat 的 McpClient(复用钉钉那套)
```

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

默认 Streamable HTTP 端点在 `http://127.0.0.1:8000/mcp`(host/port 可在 `server.py` 里调)。

## 目录结构

```
vikko_rag/
├── config.py      # 路径、模型、DeepSeek、切分/检索参数
├── embedding.py   # 本地 BGE 向量化
├── store.py       # milvus-lite 内嵌向量库(免 Docker)
├── ingest.py      # 语料切块 + 入库
├── rag.py         # 检索 + DeepSeek 生成
└── server.py      # FastMCP server
```

## 常见问题

- **pip 报 SSL 证书错误(`unable to get local issuer certificate`)**:python.org 安装的
  Python 没配 CA 证书。临时绕过:加 `--trusted-host pypi.tuna.tsinghua.edu.cn`(或
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`);永久修复:运行
  `/Applications/Python 3.12/Install Certificates.command`。
- **pip 直连 PyPI 慢 / 下 torch 卡在源码编译**:用清华镜像 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`,
  并确保钉死 `torch==2.2.2`(新版 torch 无 Intel Mac wheel,会退到源码编译)。
- **重新下载 BGE 模型**:模型在 `models/bge-small-zh-v1.5/`,是从 hf-mirror 用 curl 拉的
  (`https://hf-mirror.com/BAAI/bge-small-zh-v1.5/resolve/main/<文件>`);huggingface_hub 直连在这台机器上不稳,建议继续用 curl。
