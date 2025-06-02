import pandas as pd
from rapidfuzz import fuzz
import re

def clean_html_abstract(text):
    if pd.isna(text): return ""
    return re.sub("<[^<]+?>", "", text)

def normalize_authors(name_string):
    authors = name_string.split(",")
    return ", ".join([" ".join(name.strip().split()[::-1]) for name in authors])

def remove_duplicates(df, title_col="title", doi_col="doi"):
    seen_dois = set()
    final_rows = []

    for _, row in df.iterrows():
        doi = row[doi_col]
        title = str(row[title_col]).strip().lower()  # ✅ safer

        if pd.notna(doi):
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
            final_rows.append(row)
        else:
            duplicate = False
            for existing in final_rows:
                if fuzz.ratio(str(existing[title_col]).strip().lower(), title) > 95:
                    duplicate = True
                    break
            if not duplicate:
                final_rows.append(row)

    return pd.DataFrame(final_rows)
