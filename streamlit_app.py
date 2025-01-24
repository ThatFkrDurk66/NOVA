import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# Download NLTK data
nltk.download('punkt')

# Load YAML configuration
@st.cache
def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

config = load_yaml("self-evolving-agent-prompt-en.yaml.txt")

# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        state_dict=torch.load(model_path, map_location=torch.device("cpu")),
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer("flux_lustly-ai_v1.safetensors")

# Streamlit UI setup
st.set_page_config(page_title="NOVA Assistant", layout="wide")
st.title("NOVA Assistant")
st.markdown(config.get("description", "An advanced AI assistant."))

# User input
user_input = st.text_input("Enter your question or prompt:")

if user_input:
    with st.spinner("Processing..."):
        # Use NLTK to preprocess the input
        sentences = nltk.sent_tokenize(user_input)
        word_count = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
        
        # Display preprocessing stats
        st.write(f"Preprocessing stats: {len(sentences)} sentences, {word_count} words")

        # Generate AI response
        prompt_template = config.get("prompt_template", "{input}")
        prompt = prompt_template.replace("{input}", user_input)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Example: Use numpy for a dummy operation (e.g., scaling output length)
        response_length = len(response.split())
        scaled_length = np.sqrt(response_length)  # Example use of numpy
        st.write(f"Response length (scaled): {scaled_length:.2f}")
        
        st.subheader("AI Response:")
        st.write(response)

# Adding a sample DataFrame with Pandas
st.sidebar.header("Sample Data")
data = {
    "Input Length": [5, 10, 20],
    "Response Length": [15, 25, 35],
    "AI Confidence": [0.8, 0.9, 0.95]
}
df = pd.DataFrame(data)
st.sidebar.write("Sample DataFrame:")
st.sidebar.dataframe(df)

# Image processing with Pillow (optional)
uploaded_file = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Image Size: {img.size} (Width x Height)")
    
