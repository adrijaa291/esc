import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

nltk.download('punkt')  # Ensure nltk data is downloaded
nltk.download('stopwords')

ps = PorterStemmer()

tfidf = pickle.load(open(r"D:\AIML\sms dataset\vectorizer3.pkl", 'rb'))
model = pickle.load(open(r"D:\AIML\sms dataset\model3.pkl", 'rb'))

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
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
  #return text

st.title('Email/SMS classifier')
input_sms = st.text_input("Enter the message")

#1. Preprocess
transformed_sms = transform_text(input_sms)

#2. Vectorize
vector_input = tfidf.transform([transformed_sms])

#3. Predict
result = model.predict(vector_input)[0]

if result == 1:
    st.header("Spam")
else:
    st.header("Not spam")
