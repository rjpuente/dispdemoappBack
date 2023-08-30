from fastapi import FastAPI, File, UploadFile, Form
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import os


app = FastAPI()

# Configurar CORS para permitir solicitudes desde tu frontend (ajusta los orígenes y métodos según tus necesidades)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:19006"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'merged.pdf'

def process_doc(
    path: str = '',
    is_local: bool = False,
    question: str = ''
):
    try:
        print("la ruta es: " , path)
        print( "question: " , question)
        _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

        doc = loader.load_and_split()

        print(doc[-1])

        db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())

        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())

        respuesta = str(qa.run(question))

        return respuesta        
    
    except Exception as e:
        return {"error": str(e)}



@app.post("/upload")
async def upload_pdf(pdfFile: UploadFile = File(...), question: str = Form(...)):
    try:
        pdf_data = await pdfFile.read()

        with open("merged.pdf", "wb") as merged_pdf:
            merged_pdf.write(pdf_data)

        pdf_path = os.path.abspath("merged.pdf")

        respuesta =process_doc(
            path=pdf_path,
            is_local=True,
            question=question
        )
        print("La respuesta que se enviará es: ", respuesta)
        return {"message": respuesta}
        

    except Exception as e:
        print("Error: " + e)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)