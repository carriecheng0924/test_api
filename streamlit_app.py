from openai import OpenAI
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))

def preprocessing(text) :
    
    text = text.lower()

    text = re.sub(r"[^\w\s]" , "" , text)

    text = re.sub(r"\d+" , "" , text)

    words = text.split()

    words = [w for w in words if w not in stopwords]

    preprocessing = " ".join(words)

    return preprocessing

# set up model infos
openai_api_key = st.secrets['OPENAI_API_KEY']
models = ["ft:gpt-4o-2024-08-06:personal::B3HVAHhr", 
          "ft:gpt-4o-2024-08-06:personal::B3Sbf3WW",
         "gpt-4o-2024-08-06"]

tokenizer = AutoTokenizer.from_pretrained("carriecheng0924/test")
model = AutoModelForSequenceClassification.from_pretrained("carriecheng0924/test")

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Ask for patient information: age, gender, .
st.title(
    "Hi! Welcome to the mini Therapy AI agent! I am your therapist today."
)

st.write(
    "Let's start with playing a game. For each of the following scenario, please write a short description on what you will do or how you will feel."
)

scenario1 = st.text_input("Imagine that you are walking on the streets in a city that you have never been in, and you are trying to find a specific location, but you got lost. What would you do?")

scenario2 = st.text_input("Imagine that you originally plan to go to the movies with your friends today. Before you leave home, you discover that you forget to work on some project that would be due in 1 hour. How would you feel and what would you do?")

scenario3 = st.text_input("Imagine that you are at a keynote event listening to the speakers in a large room with many people. Suddenly, your phone ringed. How would you feel and what would you do?")


if not scenario1 or not scenario2 or not scenario3:
        st.info("Please fill in all the questions!")
else:

    # some preprocessing
    scenario1 = "I am walking on the streets in a city that I have never been in, and I am trying to find a specific location, but I got lost. " + scenario1
    scenario2 = "I originally plan to go to the movies with my friends today. Before I leave home, I discover that I forget to work on some project that would be due in 1 hour. " + scenario2
    scenario3 = "I am at a keynote event listening to the speakers in a large room with many people. Suddenly, my phone ringed. " + scenario3

    inputs = [scenario1, scenario2, scenario3]

    combine_scenarios = ""
    for s in inputs:
        combine_scenarios += s 

    # classify sentiment
    cleaned_text = preprocessing(combine_scenarios)

    inputs = tokenizer(cleaned_text , return_tensors="pt" , padding=True , truncation=True , max_length=512)
    outputs = model(**inputs)
    logists = outputs.logits  
    current_status = torch.argmax(logists , dim=1).item()

    if "status" not in st.session_state:
        st.session_state.status = current_status

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    with st.chat_message("assistant"):
        st.markdown("Hi. How can I help you today?")

    if prompt := st.chat_input("What's up?"): 

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt) 
        

        # give technique 1 if sentiment predicted to be anxiety, depression, stress
        if st.session_state.status in [0, 2, 5]:

            # Generate a response using the OpenAI API.
            stream = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:personal::B4FHWof8",
                messages=[
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient to distinguish between events, thoughts, and feelings."}
                ] +
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
        # give technique 2 if sentiment predicted to be bipolar and suicidal
        elif st.session_state.status in [1, 6]:
            # Generate a response using the OpenAI API.
            stream = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:personal::B4Fa3GDz",
                messages=[
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient explain how thoughts create feelings."}
                ] +
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
        # give technique 3 if sentiment predicted to be personality disorder and normal
        else:
            # Generate a response using the OpenAI API.
            stream = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:personal::B4G0ktRl",
                messages=[
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient look for variations in a specific belief."}
                ]+
                [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


