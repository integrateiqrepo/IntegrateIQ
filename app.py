from fastapi import FastAPI, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
import base64
from chat import (user_input, get_pdf_text, get_text_chunks, 
get_vector_store, extract_text_from_image, get_web_text, speech_to_text, text_to_speech
)
import os
import shutil
from prsnl import QA

app=FastAPI()

origins = [
    "https://titansai.agency", 
    "https://marketingtitans.web.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

context = ''

@app.post('/chat/Question')
async def get_A(Question: str = Body(...)):
    global context
    answer = user_input(Question, context)
    return {'text': answer}

@app.post('/chat/audio')
async def get_A(file: UploadFile):
    global context
    if len(context)<50:
        return {'status': False, 'msg': 'Upload Data first'}
    if file.filename.split('.')[-1] in ['wav', 'mp3', 'aac', 'm4a']:
        audio_bytes = await file.read() 
        audio_bytes_io = BytesIO(audio_bytes)
        
        text = speech_to_text(audio_bytes_io)
        answer = user_input(text, context=context)
        return {'status':True,'text':answer,'Question': text}
    else:
        return {'status':False, 'error':'Unsupported audio file type'}

@app.post('/chat/upload')
async def upload_pdf(file: UploadFile):
    raw_text = ''
    if file.filename.split('.')[-1]=='pdf':
        raw_text = get_pdf_text(file.file)
    elif file.filename.lower().split('.')[-1] in ['jpg', 'png', 'jpeg' , 'img']:
        raw_text = extract_text_from_image(file.file)
    else:
        return {'error':'Unsupported file type'}
    global context
    context = raw_text
    # text_chunks = get_text_chunks(raw_text)
    # status=get_vector_store(text_chunks)
    return {'success':True}


@app.post('/chat/url')
async def fetch_url(url: str = Body(...)):
    raw_text = get_web_text(url)
    if len(raw_text.split())<50:
        return {'success':False}
    global context
    context = raw_text
    # text_chunks = get_text_chunks(raw_text)
    # status=get_vector_store(text_chunks)
    
    return {'success':True}

@app.post('/chat/tts')
async def Tts(text: str= Body(...)):
    audio = text_to_speech(text)
    audio_base64 = base64.b64encode(audio.getvalue()).decode('utf-8')
    return JSONResponse(content={
        'status': True,
        'text': text,
        'audio_base64': audio_base64,
        'audio_format': 'wav'
    })


@app.get('/chat/clear')
async def clear():
    global context
    if os.path.exists("faiss_index") or len(context)<50:
        # shutil.rmtree("faiss_index")
        context = ''
        return {'success':True, 'msg': 'Data cleared successfully'}
    else:
        return {'success':False, 'msg':'No data to clear'}
    
@app.post('/prsnl/Question')
async def titansAi(Q:str = Body(...)):
    ans=QA(Q)
    return {'answer':ans}
    




