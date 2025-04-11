from fastapi import FastAPI, UploadFile, Form
from rag_pipeline import RAGSystem

app = FastAPI()
rag = RAGSystem()

@app.post("/upload/")
async def upload_lecture_notes(file: UploadFile):
    content = await file.read()
    text = content.decode("utf-8")
    rag.load_text_and_embed(text)
    return {"message": "Lecture notes uploaded and processed."}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    answer = rag.answer_question(question)
    return {"answer": answer}
