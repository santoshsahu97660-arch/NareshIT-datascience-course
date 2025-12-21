import streamlit as st
from PIL import Image
import os
import pathlib
import textwrap 

os.environ['GEMINI_API_KEY'] = 'AIzaSyDFFGmUpFvdP8jkF2jwE9hetFeGSTYS-9s' 

import google.generativeai as genai
genai.configure(api_key=os.environ['GEMINI_API_KEY'])


def get_gemini_response(input, image):
      model = genai.GenerativeModel("models/gemini-3-flash-preview") 
      
      if input !="":
            response = model.generate_content([input, image]) 
      else:
            response = model.generate_content([image])
            
      return response.text

st.set_page_config(page_title="Gemini-3 LLM Bootcamp by SANTOSH SAHU", page_icon="🤖", layout="wide")

st.header("🤖 Gemini-3 LLM Bootcamp by SANTOSH SAHU")
input = st.text_input('Input prompt:', key = 'input')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg","pdf"])

image = ""
if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption='Upload Image.', use_column_width=True)
      
submit = st.button('Generate Response about the image using Gemini-3 LLM') 

if submit:
      responce = get_gemini_response(input,image)
      st.subheader("Gemini-3 LLM Response:")
      st.write(responce)
                         



