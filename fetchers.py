import requests
import pandas as pd

def fetch_openalex(query, per_page=100, max_results=500):
    results = []
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "mailto": "your_email@example.com"
    }

    next_cursor = "*"
    while next_cursor and len(results) < max_results:
        params["cursor"] = next_cursor
        r = requests.get(base_url, params=params)
        data = r.json()
        results.extend(data.get("results", []))
        next_cursor = data.get("meta", {}).get("next_cursor", None)

    records = []
    for item in results:
        abstract = item.get("abstract_inverted_index")
        if abstract:
            words = sorted([(pos, word) for word, pos_list in abstract.items() for pos in pos_list])
            abstract = " ".join([w for _, w in words])
        else:
            abstract = ""

        records.append({
            "title": item.get("title"),
            "abstract": abstract,
            "authors": ", ".join([auth["author"]["display_name"] for auth in item.get("authorships", [])]),
            "year": item.get("publication_year"),
            "doi": item.get("doi"),
            "source": "OpenAlex",
            "venue": item.get("host_venue", {}).get("display_name", ""),
            "topics": ", ".join([c["display_name"] for c in item.get("concepts", [])]),
            "citations": item.get("cited_by_count")
        })
    return pd.DataFrame(records)

def fetch_crossref(query, rows=100):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    r = requests.get(url, params=params)
    data = r.json()

    records = []
    for item in data["message"]["items"]:
        year = None
        try:
            year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
        except:
            try:
                year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
            except:
                pass

        records.append({
            "title": item.get("title", [""])[0],
            "abstract": item.get("abstract", ""),
            "authors": ", ".join([f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])]) if "author" in item else "",
            "year": year,
            "doi": item.get("DOI"),
            "source": "CrossRef",
            "venue": item.get("container-title", [""])[0],
            "topics": "",
            "citations": ""
        })
    return pd.DataFrame(records)

def fetch_all(query, save_csv=True, max_results=500):
    df_openalex = fetch_openalex(query, max_results=max_results)
    df_crossref = fetch_crossref(query, rows=min(100, max_results))
    df = pd.concat([df_openalex, df_crossref], ignore_index=True)

    if save_csv:
        filename = f"literature_{query.replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename}")

    return df
