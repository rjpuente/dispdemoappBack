from fastapi import FastAPI, File, UploadFile
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os


app = FastAPI()

os.environ['OPENAI_API_KEY'] = 'personal key'
default_doc_name = 'merged.pdf'

def process_doc(
        path: str = '',
        is_local: bool = False,
        question: str = ''
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    doc = loader.load_and_split()

    print(doc[-1])

    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())

    st.write(qa.run(question))

@app.post("/upload")
async def upload_pdf(pdfFile: UploadFile = File(...), question: str = None):
    try:
        pdf_data = await pdfFile.read()

        with open("merged.pdf", "wb") as merged_pdf:
            merged_pdf.write(pdf_data)

        pdf_path = os.path.abspath("merged.pdf")

        process_doc(
            path=pdf_path,
            is_local=False,
            question=question
        )

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)