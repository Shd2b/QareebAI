from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = ""  

app = FastAPI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory="dbv1/chroma_db",
    embedding_function=embedding_model
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    retriever = db.as_retriever(search_kwargs={"k": 3})
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

الوثائق:
{context}

السؤال: {request.question}

الجواب:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}