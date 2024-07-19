import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "mistralai/mathstral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Mathstral-7B Assistant for Physics, Engineering, and Mathematics")

st.write("Enter your question related to physics, engineering, or mathematics:")
prompt = st.text_area("Question:", "Example: Calculate the force on an object with mass 10 kg and acceleration 2 m/s².")

max_length = st.sidebar.slider("Max Response Length", min_value=10, max_value=200, value=100)

if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        full_prompt = f"Assistant in physics, engineering, and mathematics: {prompt}"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Response:", response)
