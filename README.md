# 📚 RAG con Langchain + Ollama + Gemini + Chroma

Este proyecto es una demostración de un sistema RAG (Retrieval-Augmented Generation) educativo. Permite cargar documentos PDF (por ejemplo, apuntes de cursos), indexarlos con embeddings, y luego generar ejercicios prácticos usando un modelo de lenguaje como Gemini (via `langchain-google-genai`).

## 🧠 ¿Qué hace este proyecto?

1. ✅ Permite subir PDFs de cursos.
2. 🔍 Extrae y divide los contenidos en fragmentos semánticos.
3. 🧠 Indexa los contenidos usando `nomic-embed-text` y los guarda en una base vectorial con `ChromaDB`.
4. 🤖 Usa Gemini para generar ejercicios prácticos a partir de un pedido del instructor.
5. 📎 Utiliza Streamlit como interfaz de usuario.

---

## 🛠️ Requisitos

- Python 3.10+
- Ollama (para embeddings locales)
- API Key de Google para usar Gemini

---

## ⚙️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu_usuario/rag-educativo.git
cd rag-educativo
```
# Crear entorno virtual
```bash
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
```
```bash
# Instalar dependencias
pip install -r requirements.txt
```
Variables de entorno
Crea un archivo .env con tu API key de Google:
API_KEY=tu_api_key_de_google

🚀 Uso
Ejecuta la app de Streamlit:
```bash
streamlit run demo_gemini.py
```
