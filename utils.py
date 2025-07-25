import pandas as pd
from rapidfuzz import fuzz

# -----------------------------------
# ✨ Export Function for Biblioshiny
# -----------------------------------
def clean_for_biblioshiny(df):
    df_bib = df.copy()

    df_bib['Authors'] = df_bib.get('authors', "")
    df_bib['Title'] = df_bib.get('title', "")
    df_bib['Source'] = df_bib.get('journal', df_bib.get('source', ""))
    df_bib['Year'] = df_bib.get('year', "")
    df_bib['Abstract'] = df_bib.get('abstract', "")
    df_bib['DOI'] = df_bib.get('doi', "")
    df_bib['Summary'] = df_bib.get('summary', "")

    return df_bib[['Authors', 'Title', 'Source', 'Year', 'Abstract', 'DOI', 'Summary']]

# -----------------------------------
# 🔧 Cleaning Utilities
# -----------------------------------
def normalize_authors(name_string):
    if pd.isna(name_string) or not isinstance(name_string, str):
        return ""
    authors = name_string.split(",")
    return ", ".join([
        " ".join(name.strip().split()[::-1])
        for name in authors if name.strip()
    ])

def remove_duplicates(df, title_col="title", doi_col="doi"):
    seen_dois = set()
    final_rows = []

    for _, row in df.iterrows():
        doi = str(row.get(doi_col, "")).strip().lower()
        title = str(row.get(title_col, "")).strip().lower()

        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
            final_rows.append(row)
        else:
            duplicate = False
            for existing in final_rows:
                existing_title = str(existing.get(title_col, "")).strip().lower()
                if fuzz.ratio(title, existing_title) > 95:
                    duplicate = True
                    break
            if not duplicate:
                final_rows.append(row)

    return pd.DataFrame(final_rows)
