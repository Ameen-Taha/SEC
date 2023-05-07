import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import email
import uuid
import numpy as np
import joblib
import re
import string
import warnings  
warnings.filterwarnings("ignore")   

@st.cache_resource  
def load_models():
    emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    pred_model = joblib.load('model.sav')
    return emb_model,pred_model

def preprocess(x):
    # lowercasing all the words
    x = x.lower()
    
    # remove extra new lines
    x = re.sub(r'\n+', ' ', x)
    
    # removing (replacing with empty spaces actually) all the punctuations
    x = re.sub("["+string.punctuation+"]", " ", x)
    
    # remove extra white spaces
    x = re.sub(r'\s+', ' ', x)
    
    return x

emb_model,pred_model = load_models()

def predict_email_class(path):
    with open(path, 'r') as f:
        s = f.read()

    e = email.message_from_string(s)
    body = e.get_payload()
    subject = e.get('Subject')
    s = subject + ' ' + body
    s = preprocess(s)
    arr = emb_model.encode(s).ravel()
    cls = pred_model.predict([arr])[0]
    return cls


def predict_email_class2(subject,body):
    s = subject + ' ' + body
    s = preprocess(s)
    arr = emb_model.encode(s).ravel()
    cls = pred_model.predict([arr])[0]
    return cls

st.title("Email Classification Web App")
c = st.selectbox(label='select', options=['upload','type'], index=0)
file = None
if c == 'upload':
    # Create file uploader
    file = st.file_uploader("Upload a email file", type=["txt"])
else:
    sender_email = st.text_input(label='Enter sender of email')
    receiver_email = st.text_input(label='Enter receiver of email')
    subject = st.text_input(label='Enter subject of email')
    body = st.text_input(label='Enter body of email')
    cc1 = st.text_input(label='Enter cc1 of email')
    cc2 = st.text_input(label='Enter cc2 of email')
    cc3 = st.text_input(label='Enter cc3 of email')

# Save file locally
if st.button('Predict'):
    if file is not None:
        file_contents = file.read().decode("utf-8")
        with open("/content/email.txt", "w") as f:
            f.write(file_contents)
        s = file_contents
        pred = predict_email_class('/content/email.txt')
    else:
        l = [sender_email,receiver_email,subject,cc1,cc2,cc3,body]
        with open('/content/email.txt','w') as f:
            for i in l:
                if i:
                    f.write(i+'\n')
        label_names = ['sender_email','receiver_email','cc1','cc2','cc3','subject','body']
        s = ''
        for i,j in zip(l,label_names):
            if i:
                s+=j+' : '+i+'\n'
        pred = predict_email_class2(subject,body)
    with st.expander("See Input"):
        for i in s.split('\n'):
            st.write(i)
    st.success(f'Prediction from model : {pred}')
    if not os.path.isdir(f'/content/{pred}'):
        os.mkdir(f'/content/{pred}')
    os.rename('/content/email.txt',f'/content/{pred}/email-{uuid.uuid4().hex}.txt')
