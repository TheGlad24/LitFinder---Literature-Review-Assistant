import pandas as pd
from rapidfuzz import fuzz
import re

def clean_html_abstract(text):
    """
    Remove HTML tags from abstract text. Return empty string if input is not valid.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return re.sub(r"<[^<]+?>", "", text)

def normalize_authors(name_string):
    """
    Normalize author names: "Doe John" -> "John Doe"
    """
    if pd.isna(name_string) or not isinstance(name_string, str):
        return ""

    authors = name_string.split(",")
    return ", ".join([
        " ".join(name.strip().split()[::-1]) 
        for name in authors if name.strip()
    ])

def remove_duplicates(df, title_col="title", doi_col="doi"):
    """
    Remove duplicate entries based on DOI or fuzzy-matched titles.
    """
    seen_dois = set()
    final_rows = []

    for _, row in df.iterrows():
        doi = row.get(doi_col, None)
        title = str(row.get(title_col, "")).strip().lower()

        if pd.notna(doi) and isinstance(doi, str) and doi != "":
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
            final_rows.append(row)
        else:
            duplicate = False
            for existing in final_rows:
                existing_title = str(existing.get(title_col, "")).strip().lower()
                if existing_title and title and fuzz.ratio(existing_title, title) > 95:
                    duplicate = True
                    break
            if not duplicate:
                final_rows.append(row)

    return pd.DataFrame(final_rows)
