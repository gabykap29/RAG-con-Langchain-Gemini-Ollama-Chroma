import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import os
import hashlib

custom_template = """ 
Actuá como asistente para docentes en una plataforma de cursos llamada CODE-TRAIN.

Tu única tarea: redactar actividades breves y prácticas para principiantes, según el pedido del instructor.

Reglas:

1. No explicar, resumir, reformular ni agregar texto.
2. Ignorá el "Contexto" si está vacío o no es relevante.
3. No incluir ejemplos ni respuestas.
4. Siempre responder en español y en el formato indicado.

Datos:
Curso: {course_title}
Lección: {lesson_title}
Pedido: {instructor_request}
Contexto: {contexto}
Cantidad: {amount} (por defecto: 5)

Formato obligatorio:
"ejercicio1": "[texto]"
"ejercicio2": "[texto]"
...

Importante:
* Actividades simples, sin opción múltiple ni teoría.
* Dirigidas a principiantes.
* Nada extra.
"""

pdf_directory = "./data"
db_directory = "./db"

if not os.path.exists(db_directory):
    os.makedirs(db_directory)
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=db_directory,
)

def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
        
def load_pdf(file):
    loader = PDFPlumberLoader(file)
    documents = loader.load()
    return documents

def text_splitter(documents, course_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )    
    chunks = text_splitter.split_documents(documents)
    if course_name:
        for i, doc in enumerate(chunks):
            doc.metadata["course_name"] = course_name
            doc.page_content = f"Curso: {course_name}\n" + doc.page_content
    return chunks


def index_docs(documents):
    vectorstore.add_documents(documents)
    vectorstore.persist()
    print("Documents indexed successfully. Numbers of documents:", len(documents))
    

def retrieve_docs(query, course_name):
    docs = vectorstore.similarity_search(query, k=5)
    print("Retrieved documents:", len(docs))
    if course_name:
        docs = [doc for doc in docs if doc.metadata.get("course_name") == course_name]
    else:
        docs = [doc for doc in docs]
    if not docs:
        print("No documents found for the given course name.")
    return docs


def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_pdf_already_indexed(file_path):
    result = vectorstore.similarity_search(file_path, k=1)
    if result:
        for doc in result:
            if doc.metadata.get("file_hash") == get_file_hash(file_path):
                return True
    return False


from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    st.error("API key not found. Please set the API_KEY environment variable.")
    

import os

os.environ["GOOGLE_API_KEY"] = api_key


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

def generate_response_stream(context, course_tile, lesson_title, instructor_request, amount):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_tokens=2000,
        top_p=0.9,
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", custom_template),
        ("user", "{course_title} {lesson_title} {instructor_request} {contexto} {amount}"),
        ],
    )

    chain = prompt | llm | StrOutputParser()
    
    input_dict = {
        "course_title": course_title,
        "lesson_title": lesson_title,
        "instructor_request": instructor_request,
        "contexto": context,
        "amount": amount
    }
    print("Input dictionary:", input_dict)
    
    for chunk in chain.stream(input_dict):
        yield chunk 

uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
name_course = st.text_input("Nombre del curso")

if uploaded_file and name_course:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_directory + uploaded_file.name)

    file_path = pdf_directory + uploaded_file.name
    file_hash = get_file_hash(file_path)
    if is_pdf_already_indexed(file_path):

        st.warning("Este PDF ya ha sido indexado.")
    else:
        chunked_documents = text_splitter(documents, name_course)
        for doc in chunked_documents:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["course_name"] = name_course
            print("--- Documento a indexar ---")
            print(doc.page_content)
            print(doc.metadata)
        index_docs(chunked_documents)
        st.success("PDF subido y procesado correctamente.")
        
        

course_title = st.text_input("Titulo del curso")
lesson_title = st.text_input("Titulo de la lección")
instructor_request = st.text_area("Ordenes para el asistente")
amount = st.number_input("Cantidad de ejercicios a generar")


if course_title != "" and lesson_title != "" and instructor_request != "" and amount != "":
    st.chat_message("user").write(f"Título del curso: {course_title}")
    st.chat_message("user").write(f"Título de la lección: {lesson_title}")
    st.chat_message("user").write(f"Pedido del instructor: {instructor_request}")
    st.chat_message("user").write(f"Cantidad de ejercicios: {amount}")
    st.markdown("---")

  
    related_documents = retrieve_docs(lesson_title, course_title)


    contexto = "\n".join(doc.page_content for doc in related_documents) if related_documents else ""

    message_placeholder = st.chat_message("assistant").empty()
    full_response = ""

    for chunk in generate_response_stream(contexto, course_title, lesson_title ,instructor_request, amount):
        full_response += chunk  # cada chunk trae parte del texto
        message_placeholder.markdown(full_response)  
