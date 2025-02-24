# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

from openai import OpenAI
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import transformers
# from transformers import pipeline
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
    "Let's start with playing a game. For each of the following scenario, please write a short description on what you will do."
)

scenario1 = st.text_input("Imagine that you are walking on the streets in a city that you have never been in, and you are trying to find a specific location, but you got lost. What would you do?")

scenario2 = st.text_input("Imagine that you originally plan to go to the movies with your friends today. Before you leave home, you discover that you forget to work on some project that would be due in 1 hour. What would you do?")

scenario3 = st.text_input("Imagine that you are at a keynote event listening to the speakers in a large room with many people. Suddenly, your phone ringed. What would you do?")


# age_info = st.text_input("What is your age?")

# gender_info = st.text_input("What is your gender?")

# sleep_duration = st.text_input("How many hours do you sleep everyday on average? Please provide a number.")
# sleep_quality = st.text_input("From 1 to 5, please rate your sleeping quality, with 1 indicating the poorest and 5 indicating the best.")
# st.write("Do you exercise regularly?")
# has_physical_activity = st.checkbox("Yes, I exercise regularly.")
# not_has_physical_activity = st.checkbox("No, I don't exercise.")

# st.write("Do you drink alcohol?")
# is_alcohol = st.checkbox("Yes, I drink alcohol.")
# is_not_alcohol = st.checkbox("No, I don't drink alcohol.")

# med_hist_info = st.text_input("Do you have any medical conditions?")

# status_info = st.text_input("Please tell me how you are feeling today.")

# inputs = [age_info, gender_info, sleep_duration, sleep_quality, med_hist_info, status_info]
# yesorno = [(has_physical_activity, not_has_physical_activity), (is_alcohol, is_not_alcohol)]

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

    # if is_alcohol:
    #     alcohol_use = "I drink alcohol."
    # elif is_not_alcohol:
    #     alcohol_use = "I don't drink alcohol."

    # if has_physical_activity:
    #     physical_activity = "I exercise regularly."
    # elif not_has_physical_activity:
    #     physical_activity = "I don't exercise regularly."
    

    # if not status_info:
    #     st.info("Please let me know your feelings so that we could continue!")
    # else:


        # rate overall mental status given user input about living habits/medical conditions
        # st.warning(likelihood.choices[0].to_dict()['message']['content'])

    # messages = [
    #     {"role": "system", "content": "Please output only a rate from 1 to 3 about patient stress level given patient information."},
    #     {"role": "user", "content": "I am {} years old. I am {}. I usually sleep for {} hours everyday. I rate my sleeping quality as {}, where 1 means the poorest and 5 means the best. {}. {}. I am having medical conditions of {}.".format(age_info, 
    #                                                                                                                                                                                      gender_info,
    #                                                                                                                                                                                      sleep_duration,
    #                                                                                                                                                                                      sleep_quality,
    #                                                                                                                                                                                      physical_activity,
    #                                                                                                                                                                                      alcohol_use,
    #                                                                                                                                                                                     med_hist_info)},
    # ]

    # prompt = pipeline.tokenizer.apply_chat_template(
    #         messages, 
    #         tokenize=False, 
    #         add_generation_prompt=True
    # )

    # terminators = [
    #     pipeline.tokenizer.eos_token_id,
    #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    # outputs = pipeline(
    #     prompt,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     do_sample=True,
    #     temperature=0.0,
    #     top_p=0.9,
    # )

    # overall_rate = outputs[0]["generated_text"][len(prompt):]
    # st.warning(outputs[0]["generated_text"][len(prompt):])
    # actually wanted to use OpenBioLLM but model loading too slow and taking up too many memory
    # per benchmark results, GPT-4 performed close but a bit worse than OpenBioLLM but could load more quickly
    # outputs = client.chat.completions.create(
    #             model="gpt-4o-2024-08-06",
    #             messages=messages
    #         )

    # overall_rate = int(outputs.choices[0].to_dict()['message']['content'])
    # st.warning(overall_rate)

    # rate current mental status given the user input about current feelings
    cleaned_text = preprocessing(combine_scenarios)

    inputs = tokenizer(cleaned_text , return_tensors="pt" , padding=True , truncation=True , max_length=512)
    outputs = model(**inputs)
    logists = outputs.logits  
    current_status = torch.argmax(logists , dim=1).item()

    # st.warning(current_status)

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

    # st.session_state.messages.append({"role": "user", "content": status_info})

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    # if prompt := st.chat_input("What's up?"):

    #     # Store and display the current prompt.
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

        # st.warning([{"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages])

        # st.warning(st.session_state.step1)
    
    with st.chat_message("assistant"):
        st.markdown("Hi. How can I help you today?")

    if prompt := st.chat_input("What's up?"): 

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt) 
        
        # st.warning(st.session_state.rate)
        # give technique 1 if sentiment predicted to be anxiety, depression, stress
        if st.session_state.status in [0, 2, 5]:

            # Generate a response using the OpenAI API.
            stream = client.chat.completions.create(
                model="ft:gpt-4o-2024-08-06:personal::B4FHWof8",
                messages=[
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
                ] +
                # [
                #     {"role": "user", "content": status_info}
                # ] +
                # [
                #     {"role": "user", "content": stream1.choices[0].to_dict()['message']['content']}
                # ] +
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
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
                ] +
                # [
                #     {"role": "user", "content": status_info}
                # ] +
                # [
                #     {"role": "user", "content": stream1.choices[0].to_dict()['message']['content']}
                # ] +
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
                    {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
                ]+
                # [
                #     {"role": "user", "content": status_info}
                # ] +
                # [
                #     {"role": "user", "content": stream1.choices[0].to_dict()['message']['content']}
                # ] +
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


