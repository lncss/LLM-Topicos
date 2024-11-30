import os
import tempfile
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import streamlit as st

# Configuração da API Google Gemini
genai.configure(api_key="AIzaSyDrQ-mEF9CqxYkAnu5rxPFfTGIbE1O--C0")
model = genai.GenerativeModel("gemini-1.5-flash")

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Erro ao processar o PDF: {e}"

# Função para gerar embeddings usando TF-IDF
def generate_embeddings(documents):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(documents)
    return embeddings, vectorizer

# Função para responder perguntas com base no contexto
def get_response_from_model(prompt):
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):  # Verifica se o atributo 'text' está presente
            return response.text.strip()  # Retorna o texto gerado
        else:
            return "A resposta não contém um texto gerado."
    except Exception as e:
        return f"Erro ao gerar resposta com o modelo: {e}"

# Configuração da interface com Streamlit
st.title("Assistente Conversacional 📄🤖")

st.sidebar.header("Configurações")
uploaded_files = st.sidebar.file_uploader(
    "Faça upload de arquivos PDF", type=["pdf"], accept_multiple_files=True
)

# Variável para armazenar o texto extraído
all_document_texts = []

# Processar os arquivos PDF
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            document_text = extract_text_from_pdf(tmp_file.name)
            if document_text:
                all_document_texts.append(document_text)

    if all_document_texts:
        st.sidebar.success(f"{len(uploaded_files)} PDFs carregados e processados com sucesso!")
    else:
        st.sidebar.error("Erro ao processar os PDFs.")

# Caixa de texto para perguntas
if all_document_texts:
    st.write("### Faça uma pergunta:")
    question = st.text_input("Digite sua pergunta:", "")

    if question:
        # Combina os textos de todos os PDFs
        combined_text = " ".join(all_document_texts)
        documents = [combined_text]

        # Gera embeddings e encontra o texto mais relevante
        embeddings, vectorizer = generate_embeddings(documents)
        question_embedding = vectorizer.transform([question])
        similarities = cosine_similarity(question_embedding, embeddings).flatten()
        most_relevant_index = similarities.argmax()
        relevant_text = documents[most_relevant_index]

        # Gera a resposta usando o modelo
        with st.spinner("Gerando resposta..."):
            prompt = f"Baseando-se no seguinte texto: {relevant_text}\nPergunta: {question}\nResposta:"
            response = get_response_from_model(prompt)

        st.write("### Resposta:")
        st.write(response)
