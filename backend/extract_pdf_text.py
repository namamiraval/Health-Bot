# backend/extract_pdf_text.py
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

if __name__ == "__main__":
    pdf_path = "data/medical_handbook.pdf"
    text = extract_text_from_pdf(pdf_path)
    open("data/raw_text.txt", "w", encoding="utf-8").write(text)
    print("✅ Text extracted and saved to data/raw_text.txt")
