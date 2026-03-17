# %% [markdown]
# # SGK 4A Hizmet Dökümü - PDF Extractor
# Extracts TC Kimlik No and service table from SGK 4A PDF documents.

# %% Imports
import re
import pdfplumber
import pandas as pd

# %% Config
PDF_PATH = "Sosyal_Güvenlik_Kurumu_-_4A_Hizmet_Dökümü__Son_6_ay_.pdf"

# %% Extract TC Kimlik No
def extract_tc_kimlik(pdf_path: str) -> str | None:
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()
    match = re.search(r"T\.C\.\s*Kimlik\s*No\s+(\d{11})", text)
    return match.group(1) if match else None

tc_kimlik = extract_tc_kimlik(PDF_PATH)
print(f"TC Kimlik No: {tc_kimlik}")

# %% Extract service table
def extract_hizmet_table(pdf_path: str) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        table = pdf.pages[0].extract_tables()[0]

    # Row 0 is title, row 1 is header
    headers = [h.replace("\n", " ") for h in table[1]]
    rows = []
    for row in table[2:]:
        # Skip yearly summary rows (have None in most columns)
        if row[1] is None:
            continue
        cleaned = [c.replace("\n", " ") if c else "" for c in row]
        rows.append(cleaned)

    df = pd.DataFrame(rows, columns=headers)

    # Type conversions
    df["Gün"] = df["Gün"].astype(int)
    df["P.E.K/ Bsmk Dğr /Ek Gstrg"] = (
        df["P.E.K/ Bsmk Dğr /Ek Gstrg"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    return df

df = extract_hizmet_table(PDF_PATH)
print(df.to_string(index=False))

# %% Quick summary
print(f"\nToplam gün: {df['Gün'].sum()}")
print(f"Toplam kazanç: {df['P.E.K/ Bsmk Dğr /Ek Gstrg'].sum():,.2f} TL")
