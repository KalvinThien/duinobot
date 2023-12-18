import streamlit as st
import google.generativeai as genai
from fpdf import FPDF
from pdfminer.high_level import extract_text

# Hardcoded API key (replace "YOUR_API_KEY" with your actual API key)
api_key = "AIzaSyD6pimrC1BlP1rbcE6smbIYfvdP2LvbBBY"

# Streamlit page configuration
st.set_page_config(page_title="Generative Chatbot", layout="wide", initial_sidebar_state="collapsed")

# Streamlit app title and sidebar
st.title("🤖 Welcome To DuinoBot 🤖")

# Navigation bar
def navigation():
    return st.sidebar.radio("Navigation", ["Chat", "PDF Chat", "Image Chat", "Settings"])

# Set up the model configuration
def configure_model():
    return {
        "temperature": st.sidebar.slider("Temperature", 0.0, 1.0, 0.4),
        "top_p": st.sidebar.slider("Top-p", 0.0, 1.0, 1.0),
        "top_k": st.sidebar.slider("Top-k", 1, 32, 32),
        "max_output_tokens": st.sidebar.slider("Max Output Tokens", 100, 951357, 4096),
    }

def safety_settings():
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

conversation_config = [
    {"role": "user", "parts": "who are you?"},
    {"role": "model", "parts": "I am DuinoBot, a large language model trained by Jalal Mansour."},
    # ... Add more conversation parts as needed
]

# Set up the model configuration for image chat
def configure_image_model():
    return {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 10096,
    }

# Collect dynamic prompt parts from the user
def collect_user_input(prompt_input_key, generation_config, chat_history, nav_choice, conversation_config):
    while True:
        user_input = st.text_input("You:", key=f"user_input_{prompt_input_key}")
        st.text(f"user: {user_input}")

        if not user_input:
            break

        chat_history.append(("user", user_input))
        model_name = "gemini-pro"
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings(),
        )

        genai.configure(api_key=api_key)
        response = model.generate_content([history_item[1] for history_item in chat_history])
        response_text = response.parts[0].text if response.parts else ""
        chat_history.append(("DuinoBot", response_text))
        st.text(f"DuinoBot: {response_text}")
        prompt_input_key += 1

    return chat_history

# Function to read from a log file
def read_from_log_file(file_content, generation_config):
    responses = process_prompts([file_content], generation_config)
    return responses

def handle_chat_log(generation_config, chat_history):
    show_chat_log = st.checkbox("Show Chat Log")

    with st.expander("DuinoBot:"):
        uploaded_file = st.file_uploader("Upload a chat file (txt):", type=["txt"])

        if uploaded_file is not None:
            uploaded_content = uploaded_file.read().decode("utf-8")
            st.text("Uploaded Chat:")
            st.text(uploaded_content)
            responses = read_from_log_file(uploaded_content, generation_config)

            for response_text in responses:
                st.text(f"DuinoBot: {response_text}")

            chat_history.extend([(role, message) for role, message in zip(["user"] * len([uploaded_content]),
                                                                         [uploaded_content])])
            chat_history.extend([(role, message) for role, message in zip(["AI"] * len(responses), responses)])

        if show_chat_log:
            for role, message in chat_history:
                st.text(f"{role}: {message}")

            if st.button("Save Now"):
                download_content = "\n".join([f"{role}: {message}" for role, message in chat_history])
                download_filename = "chat_logs.txt"
                st.download_button(label="Download", data=download_content, file_name=download_filename)

def display_chat_history(prompt_input_key, nav_choice, generation_config, chat_history, conversation_config):
    if nav_choice == "Chat":
        show_chat_log = st.checkbox("Show Chat Log")

        with st.expander("DuinoBot:"):
            if show_chat_log:
                for role, message in chat_history:
                    st.text(f"{role}: {message}")

                if st.button("Save Chat Logs"):
                    download_content = "\n".join([f"{role}: {message}" for role, message in chat_history])
                    download_filename = "chat_logs.txt"
                    st.download_button(label="Download Chat Logs", data=download_content, file_name=download_filename)

                    # Export chat logs to PDF when the button is clicked
                    export_chat_logs_to_pdf(chat_history)

    elif nav_choice == "PDF Chat":
        pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if pdf_file is not None:
          pdf_text = extract_text(pdf_file)
        st.text("PDF Text:")
        st.text(pdf_text)

        # Text input for user to interact with PDF content
        pdf_interaction_input = st.text_input("Interact with PDF Content:")
        st.text(f"user: {pdf_interaction_input}")

        # Include user's input related to PDF content in chat history
        chat_history.append(("user", pdf_interaction_input))
        chat_history.append(("user", pdf_text))  # Include PDF text in chat history
        model_name = "gemini-pro"
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings(),
        )

        genai.configure(api_key=api_key)
        response = model.generate_content([history_item[1] for history_item in chat_history])
        response_text = response.parts[0].text if response.parts else ""
        chat_history.append(("DuinoBot", response_text))
        st.text(f"DuinoBot: {response_text}")

    elif nav_choice == "Image Chat":
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])
        generation_config = configure_image_model()

        # Input field to write a prompt based on the image

        # Default prompt after uploading the image
        default_prompt = "What do you see in this image?"
        resized_image_width = 777
        # Trigger content generation when the user hits Enter
        image_prompt = st.text_input("Write a prompt based on the image:")
        # Button to generate content based on the prompt
        if st.button("Run"):
            st.text(f"user: {image_prompt}")
            # Use the provided prompt or the default prompt if none is provided
            prompt_to_use = image_prompt.strip() if image_prompt else default_prompt
            # ... existing code for image preparation (resized_image_width)

            st.image(uploaded_file, caption="Uploaded Image.", width=resized_image_width, use_column_width=2048, )
            # Process image and generate content based on the prompt
            responses = process_image_upload(uploaded_file, generation_config, prompt_to_use)

            # Display chat history for the current prompt
            st.subheader("Chat History:")
            for i, response_text in enumerate(responses):
                st.text(f"DuinoBot: {response_text}")

                # Add a button to download the chat logs for the current prompt
                download_content = "\n".join([f"DuinoBot: {response_text}" for response_text in responses])
                download_filename = f"chat_logs_prompt_{i}.txt"
                st.download_button(label=f"Save ({i + 1})", data=download_content, file_name=download_filename)

            # Add a button to export chat history to PDF
            if st.button("Export Chat History to PDF"):
                export_chat_logs_to_pdf(chat_history)

            # Resize the displayed image with a specific width (adjust as needed)
            # You can adjust the width based on your preference

            # Display the uploaded image with hover effect

    elif nav_choice == "Settings":
        st.subheader("Model Settings:")
        st.write(generation_config)

def process_image_upload(uploaded_file, generation_config, prompt):
    model_name = "gemini-pro-vision"
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings(),
    )

    genai.configure(api_key=api_key)

    if uploaded_file:
        image_bytes = uploaded_file.read()
        image_parts = [{"mime_type": "image/png", "data": image_bytes}]
        prompt_parts = [image_parts[0], prompt]

        # Generate content
        response = model.generate_content(prompt_parts)
        generated_text = response.text

        # Display generated content
        st.subheader("Duino 👁️:")
        st.write(generated_text)

        # Display the uploaded image
        st.subheader("Uploaded Image:")
        # st.image(image_bytes, caption="Uploaded Image.", use_column_width=300)

        # Return a list containing the generated text
        return [generated_text]
    else:
        return []  # Return an empty list if no file is uploaded

# Main part of the Streamlit app
nav_choice = navigation()
generation_config = configure_model()
chat_history = []

if nav_choice == "Chat":
    collect_user_input(0, generation_config, chat_history, nav_choice, conversation_config)
    handle_chat_log(generation_config, chat_history)

elif nav_choice == "PDF Chat":
    display_chat_history(0, nav_choice, generation_config, chat_history, conversation_config)

elif nav_choice == "Image Chat":
    display_chat_history(0, nav_choice, generation_config, chat_history, conversation_config)

elif nav_choice == "Settings":
    st.subheader("Model Settings:")
    st.write(generation_config)