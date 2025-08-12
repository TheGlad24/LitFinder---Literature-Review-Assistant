# fetchers.py
import requests
import pandas as pd
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Helpers
# -----------------------------

def reconstruct_abstract(index: dict) -> str:
    """Rebuild OpenAlex's abstract from inverted index."""
    if not isinstance(index, dict):
        return ""
    try:
        words = sorted(index.items(), key=lambda x: min(x[1]))
        return " ".join([w for w, _ in words])
    except Exception:
        return ""

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "LitFinder/1.0 (+https://litfinder.app)"})
    return s

# -----------------------------
# OpenAlex
# -----------------------------

def fetch_openalex(query: str, max_results: int = 500) -> List[Dict[str, Any]]:
    """Paginate OpenAlex until max_results or cursor is exhausted."""
    url = "https://api.openalex.org/works"
    per_page = min(200, max_results)
    params = {
        "search": query,
        "per_page": per_page,
        "cursor": "*",
        "select": ",".join([
            "title", "authorships", "publication_year",
            "abstract_inverted_index", "host_venue", "doi",
        ]),
        # add filters=... if you want to restrict types/years
    }
    sess = _session()
    results: List[Dict[str, Any]] = []

    while len(results) < max_results:
        try:
            resp = sess.get(url, params=params, timeout=(3, 15))
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", "") or "",
                    "authors": ", ".join([
                        (a.get("author", {}) or {}).get("display_name", "")
                        for a in item.get("authorships", []) if a
                    ]).strip(", "),
                    "year": item.get("publication_year", ""),
                    "abstract": reconstruct_abstract(item.get("abstract_inverted_index", {})),
                    "journal": (item.get("host_venue", {}) or {}).get("display_name", "") or "",
                    "doi": item.get("doi", "") or "",
                    "source": "openalex",
                })
                if len(results) >= max_results:
                    break

            next_cursor = (data.get("meta", {}) or {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
        except Exception:
            break

    return results

# -----------------------------
# Crossref
# -----------------------------

def fetch_crossref(query: str, max_results: int = 500) -> List[Dict[str, Any]]:
    """Fetch up to max_results from Crossref (rows is capped ~1000)."""
    url = "https://api.crossref.org/works"
    rows = min(1000, max_results)
    params = {
        "query": query,
        "rows": rows,
        "select": ",".join(["title", "author", "issued", "abstract", "container-title", "DOI"]),
    }
    sess = _session()
    results: List[Dict[str, Any]] = []
    try:
        resp = sess.get(url, params=params, timeout=(3, 15))
        resp.raise_for_status()
        data = resp.json()
        for item in (data.get("message", {}) or {}).get("items", []):
            title_field = item.get("title", "")
            title = title_field[0] if isinstance(title_field, list) and title_field else (title_field or "")

            authors = []
            for a in item.get("author", []) or []:
                name = " ".join(filter(None, [a.get("given"), a.get("family")]))
                if name:
                    authors.append(name)

            issued = item.get("issued", {}) or {}
            year = None
            if isinstance(issued.get("date-parts"), list) and issued["date-parts"] and isinstance(issued["date-parts"][0], list):
                year = issued["date-parts"][0][0]

            abstract = (item.get("abstract") or "")
            abstract = abstract.replace("<jats:p>", "").replace("</jats:p>", "")

            container = item.get("container-title", "")
            journal = container[0] if isinstance(container, list) and container else (container or "")

            results.append({
                "title": title,
                "authors": ", ".join(authors),
                "year": year,
                "abstract": abstract,
                "journal": journal,
                "doi": item.get("DOI", "") or "",
                "source": "crossref",
            })
            if len(results) >= max_results:
                break
    except Exception:
        pass
    return results

# -----------------------------
# arXiv (optional)
# -----------------------------

def fetch_arxiv(query: str, max_results: int = 300) -> List[Dict[str, Any]]:
    """
    Fetch from arXiv Atom feed. Requires `feedparser` (pip install feedparser).
    If feedparser isn't installed, returns [] gracefully.
    """
    try:
        import feedparser  # type: ignore
    except Exception:
        return []

    base = "http://export.arxiv.org/api/query"
    per_call = 100
    fetched: List[Dict[str, Any]] = []
    start = 0
    sess = _session()

    while len(fetched) < max_results:
        n = min(per_call, max_results - len(fetched))
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": n,
            "sortBy": "relevance",
        }
        url = base + "?" + "&".join([f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()])
        try:
            # arXiv prefers a polite user-agent; we set that in _session()
            resp = sess.get(url, timeout=(3, 15))
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if not feed.entries:
                break
            for e in feed.entries:
                authors = ", ".join([a.get("name", "") for a in e.get("authors", [])])
                abstract = (e.get("summary", "") or "").strip()
                journal = (e.get("arxiv_journal_ref") or "")
                year = None
                if e.get("published_parsed"):
                    year = e.published_parsed.tm_year
                fetched.append({
                    "title": (e.get("title", "") or "").replace("\n", " ").strip(),
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "journal": journal,
                    "doi": "",
                    "source": "arxiv",
                })
                if len(fetched) >= max_results:
                    break
            start += n
        except Exception:
            break
    return fetched

# -----------------------------
# Orchestrator
# -----------------------------

def fetch_all(
    query: str,
    max_results: int = 500,
    save_csv: bool = False,
    sources: Tuple[str, ...] = ("openalex", "crossref"),
) -> pd.DataFrame:
    """Fetch from multiple sources in parallel and return a unified DataFrame."""
    tasks = []
    with ThreadPoolExecutor(max_workers=len(sources)) as ex:
        for src in sources:
            if src == "openalex":
                tasks.append(ex.submit(fetch_openalex, query, max_results))
            elif src == "crossref":
                tasks.append(ex.submit(fetch_crossref, query, max_results))
            elif src == "arxiv":
                tasks.append(ex.submit(fetch_arxiv, query, max_results))

        frames = []
        for f in as_completed(tasks):
            try:
                data = f.result()
                if data:
                    frames.append(pd.DataFrame(data))
            except Exception:
                continue

    if not frames:
        raise ValueError("No data returned from any source.")

    df = pd.concat(frames, ignore_index=True)

    # Light standardization
    keep_cols = ["title", "authors", "year", "abstract", "journal", "doi", "source"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[keep_cols]

    if save_csv:
        df.to_csv("fetched_data.csv", index=False)

    return df
