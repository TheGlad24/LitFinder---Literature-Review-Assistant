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

import requests

def fetch_crossref(query, rows=100):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # raises HTTPError if status not 200
        data = response.json()

        # Safely access nested keys
        items = data.get("message", {}).get("items", [])
        if not items:
            print("No results returned from Crossref for query:", query)

        results = []
        for item in items:
            results.append({
                "title": item.get("title", [""])[0],
                "authors": ", ".join([
                    f"{a.get('family', '')} {a.get('given', '')}" 
                    for a in item.get("author", [])
                ]) if "author" in item else "",
                "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                "abstract": item.get("abstract", ""),
                "journal": item.get("container-title", [""])[0]
            })

        return results

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def fetch_all(query, save_csv=False, max_results=500):
    print(f"Fetching for query: {query}")
    
    data_sources = []

    df_openalex = fetch_openalex(query, rows=min(100, max_results))
    if df_openalex:
        df_openalex = pd.DataFrame(df_openalex)
        data_sources.append(df_openalex)

    df_crossref = fetch_crossref(query, rows=min(100, max_results))
    if df_crossref:
        df_crossref = pd.DataFrame(df_crossref)
        data_sources.append(df_crossref)

    if not data_sources:
        raise ValueError("No data returned from any source.")

    df = pd.concat(data_sources, ignore_index=True)

    if save_csv:
        df.to_csv("fetched_data.csv", index=False)

    return df
