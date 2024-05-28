import streamlit as st
import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# preprocess
def transform_text(text):
    text = text.lower() # converting all the chars to lowercase
    text = nltk.word_tokenize(text) # tokenization
    y=[] # removing special chars
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS classifier')

input_sms = st.text_input('Enter the msg')

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    print(model.predict(vector_input))
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')


