"""定时爬取量子位最新文章:抓首页 → 解析 → 按 ID 去重 → 广告过滤 → 增量入库。"""
import html
import re

import requests

from . import config, ingest
from .ad_filter import is_ad

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
}


def fetch_article_ids(limit: int = config.CRAWL_COUNT) -> list[str]:
    """抓首页,按文章 ID 从大到小返回最新 limit 篇的 URL。"""
    resp = requests.get(config.QBITAI_HOME, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    urls = re.findall(r"https://www\.qbitai\.com/\d{4}/\d{2}/\d+\.html", resp.text)
    # 去重保序
    seen, unique = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    # 按 ID 从大到小(最新在前)
    unique.sort(key=lambda u: int(u.rstrip(".html").rsplit("/", 1)[1]), reverse=True)
    return unique[:limit]


def fetch_article(url: str) -> tuple[str, str]:
    """抓单篇,返回 (标题, 正文)。"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    text = resp.text
    m = re.search(r'property="og:title" content="([^"]*)"', text)
    title = html.unescape(m.group(1)).strip() if m else ""
    m = re.search(r'<div class="content">(.*?)<div class="content_right">', text, re.S)
    body = m.group(1) if m else ""
    body = re.sub(r"</(p|h1|h2|h3|h4|li|blockquote|div)>", "\n", body)
    body = re.sub(r"<br\s*/?>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body).replace("扫码关注量子位", "")
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    return title, "\n\n".join(lines)


def crawl() -> int:
    """跑一轮爬取:去重 → 广告过滤 → 存 data/ → 增量入库。返回新增篇数。"""
    new_files = []
    for url in fetch_article_ids():
        aid = url.rstrip(".html").rsplit("/", 1)[1]
        if (config.DATA_DIR / f"{aid}.md").exists():
            continue  # 去重:已抓过
        title, body = fetch_article(url)
        if not body:
            continue
        if is_ad(title, body):
            print(f"⏭️  跳过广告: {title}")
            continue
        path = config.DATA_DIR / f"{aid}.md"
        path.write_text(f"# {title}\n\n来源: {url}\n\n{body}\n", encoding="utf-8")
        print(f"✅ 新增: {title}")
        new_files.append(path)

    if new_files:
        # 幂等 upsert;re-ingest 全部,顺带兜底「上次存了文件但没入库」的遗漏
        ingest.ingest()
    print(f"本轮完成:新增 {len(new_files)} 篇")
    return len(new_files)


if __name__ == "__main__":
    crawl()
