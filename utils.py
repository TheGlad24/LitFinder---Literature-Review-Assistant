# --- File: utils.py ---
import pandas as pd
from rapidfuzz import fuzz

# -----------------------------------
# ✨ Export Function for Biblioshiny
# -----------------------------------
def clean_for_biblioshiny(df):
    df_bib = df.copy()
    df_bib['Authors'] = df_bib['authors']
    df_bib['Title'] = df_bib['title']
    df_bib['Source'] = df_bib['source']
    df_bib['Year'] = df_bib['year']
    df_bib['Abstract'] = df_bib['abstract']
    df_bib['DOI'] = df_bib['doi']
    df_bib['Summary'] = df_bib.get('summary', "")  # safely handle missing summaries

    return df_bib[['Authors', 'Title', 'Source', 'Year', 'Abstract', 'DOI', 'Summary']]

# -----------------------------------
# 🔧 Cleaning Utilities
# -----------------------------------
def normalize_authors(name_string):
    authors = name_string.split(",")
    return ", ".join([" ".join(name.strip().split()[::-1]) for name in authors])

def remove_duplicates(df, title_col="title", doi_col="doi"):
    seen_dois = set()
    final_rows = []

    for _, row in df.iterrows():
        doi = row[doi_col]
        title = row[title_col].strip().lower()
        if pd.notna(doi):
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
            final_rows.append(row)
        else:
            # Check title similarity (if no DOI)
            duplicate = False
            for existing in final_rows:
                if fuzz.ratio(existing[title_col].strip().lower(), title) > 95:
                    duplicate = True
                    break
            if not duplicate:
                final_rows.append(row)

    return pd.DataFrame(final_rows)
