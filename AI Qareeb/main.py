from xml.dom.minidom import Document
from fastapi import FastAPI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"   

app = FastAPI()

PERSIST_DIR = "dbv1/chroma_db"

# تحميل ملفات الاسعافات الأولية
def load_txt_files(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    print(f"✅ Loaded {len(documents)} files")
    return documents


# إعداد embedding 
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# بناء أو تحميل DB
if not os.path.exists(PERSIST_DIR):
    print("Building DB...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCS_PATH = os.path.join(BASE_DIR, "docs")

    documents = load_txt_files(DOCS_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_text(doc))

    print(f"Created {len(chunks)} chunks")

    processed_chunks = [
        Document(page_content=chunk, metadata={"source": "txt"})
        for chunk in chunks
    ]

    db = Chroma.from_documents(
        documents=processed_chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )

    print("DB created successfully!")

else:
    print("Loading existing DB...")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in results])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""أنت مساعد إسعافات أولية متخصص.
اعتمد فقط على المعلومات الموجودة في الوثائق.

القواعد:
- أجب باللغة العربية فقط
- استخدم خطوات مرقمة
- كن واضحاً ومباشراً
- لا تضف معلومات من خارج البيانات
- اذا لم تذكر الفئة العمرية في البيانات قدم الاسعافات لجميع الفئات العمرية

الوثائق:
{context}

السؤال: {request.question}

الجواب:"""
    response = llm.invoke(prompt)
    return {"answer": response.content}