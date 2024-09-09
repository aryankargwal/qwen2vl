import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import requests
from io import BytesIO

# Load the model and processor
@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

model, processor = load_model()

# Streamlit UI
st.title("Image Description Generator with Qwen2-VL")
st.write("Upload an image and the model will describe it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to a format suitable for the model
    image.save("uploaded_image.jpg")
    
    # Prepare input for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "uploaded_image.jpg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to GPU (if available)
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inference: Generate description
    with st.spinner('Generating description...'):
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    # Display the output
    st.subheader("Generated Description")
    st.write(output_text[0])
