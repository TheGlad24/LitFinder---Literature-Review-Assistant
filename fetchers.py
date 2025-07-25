import requests
import pandas as pd


def reconstruct_abstract(inverted_index):
    """
    Convert OpenAlex's abstract_inverted_index dict to a plain abstract string.
    """
    if not isinstance(inverted_index, dict):
        return ""
    
    words = sorted([(pos, word) for word, positions in inverted_index.items() for pos in positions])
    words.sort()  # ensure sorted by position
    return " ".join(word for _, word in words)


def fetch_openalex(query, max_results=500):
    url = "https://api.openalex.org/works"
    per_page = 200  # Max allowed by OpenAlex
    params = {
        "search": query,
        "per-page": per_page,
        "cursor": "*"
    }

    results = []
    total_fetched = 0

    while total_fetched < max_results:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
                results.append({
                    "title": item.get("title", ""),
                    "authors": ", ".join([
                        auth.get("author", {}).get("display_name", "")
                        for auth in item.get("authorships", [])
                    ]),
                    "year": item.get("publication_year", ""),
                    "abstract": abstract,
                    "journal": item.get("host_venue", {}).get("display_name", ""),
                    "doi": item.get("doi", "")
                })
                total_fetched += 1
                if total_fetched >= max_results:
                    break

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor

        except Exception as e:
            print(f"Error fetching OpenAlex data: {e}")
            break

    return results


def fetch_crossref(query, rows=100):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

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
                "journal": item.get("container-title", [""])[0],
                "doi": item.get("DOI", "")
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

    # Limit each source to 100 for balance if total max > 100
    per_source = min(100, max_results)

    df_openalex = fetch_openalex(query, max_results=per_source)
    if df_openalex:
        df_openalex = pd.DataFrame(df_openalex)
        data_sources.append(df_openalex)

    df_crossref = fetch_crossref(query, rows=per_source)
    if df_crossref:
        df_crossref = pd.DataFrame(df_crossref)
        data_sources.append(df_crossref)

    if not data_sources:
        raise ValueError("No data returned from any source.")

    df = pd.concat(data_sources, ignore_index=True)

    if save_csv:
        df.to_csv("fetched_data.csv", index=False)

    return df
