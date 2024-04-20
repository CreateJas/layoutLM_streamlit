import streamlit as st
from transformers import pipeline
from PIL import Image
import io

# Initialize the document QA pipeline
nlp = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

def process_image(file, question):
    image = Image.open(io.BytesIO(file.getvalue()))
    response = nlp(image, question)
    return response

st.title('Document Question Answering System')

# Option for single or multiple file uploads
upload_option = st.radio("Choose upload mode:", ('Single File', 'Multiple Files'))

if upload_option == 'Single File':
    uploaded_file = st.file_uploader("Upload your document image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    files_to_process = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload your document images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    files_to_process = uploaded_files

# Option for single or multiple questions
question_option = st.radio("Do you want to use one question for all images or individual questions for each image?", ('One Question for All', 'Individual Questions'))

questions = []
if question_option == 'One Question for All':
    question = st.text_input("Enter your question", "What is the total amount?")
    questions = [question] * len(files_to_process)  # Same question for all files
elif question_option == 'Individual Questions' and files_to_process:
    for file in files_to_process:
        questions.append(st.text_input(f"Enter your question for {file.name}", "What is the total amount?"))

if st.button('Analyze'):
    if files_to_process and all(questions):
        for file, question in zip(files_to_process, questions):
            with st.spinner(f'Processing {file.name}...'):
                answer = process_image(file, question)
                if answer:
                    st.write(f"Answer for {file.name}:", answer[0]['answer'])
                else:
                    st.error(f"No answer found in {file.name}.")
    else:
        st.error("Please upload at least one image file and ensure all questions are entered.")

