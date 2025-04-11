from fastapi import FastAPI, UploadFile, Form
from rag_pipeline import RAGSystem
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os

app = FastAPI()
rag = RAGSystem()

@app.post("/upload/")
async def upload_lecture_notes(file: UploadFile):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    content = await file.read()

    # Save temp PDF
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(content)

    # Extract full text (text + OCR fallback)
    full_text = extract_text_from_pdf(temp_path)

    # Load into RAG
    rag.load_text_and_embed(full_text)

    os.remove(temp_path)  # cleanup
    return {"message": "Lecture notes uploaded and processed from PDF."}


@app.post("/ask/") # User Query
async def ask_question(question: str = Form(...)):
    answer = rag.answer_question(question)
    return {"answer": answer}


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    
    # Load with PyMuPDF
    pdf_doc = fitz.open(pdf_path)

    print(f"Processing {len(pdf_doc)} pages...")
    for i, page in enumerate(pdf_doc):
        text = page.get_text()
        if len(text.strip()) > 10:
            full_text += f"\n\n[Page {i+1} - Text]\n{text}"
        else:
            print(f"Fallback to OCR on page {i+1}")
            # OCR fallback
            with tempfile.TemporaryDirectory() as path:
                images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, output_folder=path)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    full_text += f"\n\n[Page {i+1} - OCR]\n{ocr_text}"
    
    pdf_doc.close()
    return full_text