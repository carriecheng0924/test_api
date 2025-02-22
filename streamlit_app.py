# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

from openai import OpenAI
import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

openai_api_key = st.secrets['OPENAI_API_KEY']
models = ["ft:gpt-4o-2024-08-06:personal::B3HVAHhr", 
          "ft:gpt-4o-2024-08-06:personal::B3Sbf3WW",
         "gpt-4o-2024-08-06"]
# with st.sidebar:
#     st.title('🤖💬 OpenAI Chatbot')
#     if 'OPENAI_API_KEY' in st.secrets:
#         st.success('API key already provided!', icon='✅')
#         openai.api_key = st.secrets['OPENAI_API_KEY']
#     else:
#         openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
#         if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
#             st.warning('Please enter your credentials!', icon='⚠️')
#         else:
#             st.success('Proceed to entering your prompt message!', icon='👉')

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What's up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.warning([{"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages])

    # sample from 3 models and use the reward function to determine which one could yield the largest reward
    if len(st.session_state.messages) == 1:
        # step1 = client.chat.completions.create(
        #     model="ft:gpt-4o-2024-08-06:personal::B3Wn4Vw9",
        #     messages= [
        #         {"role": "system", "content": "You are a therapist that decides which therapy strategy to treat the patient."}
        #     ] +
        #     [
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ]
        # )

        samples = []
        for m in models:
            stream = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
            ] +
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            # stream=True,
        )
            samples += [stream]

        prompt_content = zip(st.session_state.messages * len(models), samples)
        
        tokenizer = AutoTokenizer.from_pretrained("carriecheng0924/test")
        inputs = tokenizer([["I need a mental therapy." + message["content"], [sample]] for message, sample in prompt_content], return_tensors="pt", padding=True)
        labels = torch.tensor(0).unsqueeze(0)
        model = AutoModelForMultipleChoice.from_pretrained("carriecheng0924/test")
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        # To store value of first step
        if "step1" not in st.session_state:
            # st.session_state.step1 = step1.choices[0].to_dict()['message']['content']
            st.session_state.step1 = predicted_class

    st.warning(st.session_state.step1)
    if st.session_state.step1 == 0:

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:personal::B3HVAHhr",
            messages=[
                {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
            ] +
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
    elif st.session_state.step1 == 1:
        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="ft:gpt-4o-2024-08-06:personal::B3Sbf3WW",
            messages=[
                {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
            ] +
            [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

    else:
        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a therapist to address patient emotions. You should help patient understand his emotional and mental status."}
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
