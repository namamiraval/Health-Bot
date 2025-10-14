# backend/extract_pdf_text.py
# Step 1: Data Exploration

import pandas as pd

df = pd.read_csv('Dipiro-Handbook.pdf')

import fitz  # PyMuPDF

def extract_text_from_pdf(df):
    text = ""
    with fitz.open(df) as doc:
        for page in doc:
            text += page.get_text()
    return text

if __name__ == "__main__":
    #pdf_path = "data/medical_handbook.pdf"
    text = extract_text_from_pdf(df)
    open("raw_text.txt", "w", encoding="utf-8").write(text)
    print("Text extracted and saved to data/raw_text.txt")

