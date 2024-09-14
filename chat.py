from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import io
from PIL import Image
import pytesseract
import speech_recognition as sr
from gtts import gTTS
import os
from io import BytesIO
from urllib.parse import urljoin
from langchain.schema import Document

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def speech_to_text(audio_bytes_io):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_bytes_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return None


def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp


def get_pdf_text(pdf):
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text

def get_web_text(link):
    try:
        req = requests.get(link)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, 'html.parser')
        main_text = soup.get_text().lower()
        anchors = soup.find_all('a', href=True)
        all_texts = [main_text]
        n=10
        for anchor in anchors:
            if n==0:
                break
            n-=1
            href = anchor['href']
            
            full_url = urljoin(link, href)
            
            try:
                sub_req = requests.get(full_url)
                sub_req.raise_for_status()
                sub_soup = BeautifulSoup(sub_req.content, 'html.parser')
                sub_text = sub_soup.get_text().lower()
                all_texts.append(sub_text)
            except requests.exceptions.RequestException as e:
                continue  
        
        all_text = ' '.join(all_texts)
        temp_text =''
        x=' '
        n=5000
        for c in all_text:
            if c==x and (x==' ' or x=='\n'):
                continue
            if c==' ':
                n-=1
            if n==0:
                break
            if c=='\n':
                c=' '
            temp_text+=c
            x=c
        return temp_text
    
    except requests.exceptions.RequestException as e:
        return ''
    except Exception as e:
        return ''

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    if vector_store:
        return True
    else:
        return False


def get_conversational_chain():

    prompt_template = """
    You are acting as the chatbot for the company described in the context. Follow these rules:

    Fetch the company's name from the context and use it naturally in your responses without explicitly stating that you're the "official chatbot" in every reply.
    Answer questions based on the company's services as described in the context, while maintaining a professional and concise tone.
    If the context remains the same across multiple questions, maintain the flow of the conversation, ensuring consistency with previous responses.
    If the question is unrelated to the company or asks for a service the company does not provide, politely explain that without providing incorrect or misleading information.
    Keep responses to two lines or less, ensuring they are helpful and directly relevant to the user's inquiry.
    For casual remarks (e.g., "Hi", "Thanks"), respond in a friendly, engaging tone.
    Your name is "Smart-Chat", but do not explicitly mention this unless it adds value to the response.
    Context:

    {context}

    Question:

    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, context):

    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    docs = [Document(page_content=context, metadata={})]
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    out=response["output_text"]
    return out
    
    
    

def extract_text_from_image(image):
    file_bytes = image.read()
    image = Image.open(io.BytesIO(file_bytes))
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text