import streamlit as st
import requests

st.title('AI Communication Dashboard')
text = st.text_area('Enter your message:')

if st.button('Analyze'):
    res = requests.post('http://localhost:8000/analyze', json={'text': text})
    st.json(res.json())