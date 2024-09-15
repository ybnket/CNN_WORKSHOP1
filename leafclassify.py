import streamlit as st
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import *
from keras import preprocessing
import time
import base64
## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##


# fig = plt.figure()

st.markdown("<h1 style='color: black;'>Cassava Disease Prediction</h1>", unsafe_allow_html=True)

#st.markdown("Prediction Platform")
def set_background(main_bg):  # local image
    # set bg name
    main_bg_ext = "png"
    st.markdown(
        f"""
             <style>
             .stApp {{
                 background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                 background-size: cover
             }}
             </style>
             """,
        unsafe_allow_html=True
    )


set_background('cassava.png')
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=False)
    class_btn = st.button("Detect")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('Results')
                st.write(predictions)


# Load the .h5 model (Keras SavedModel format)
model = tf.keras.models.load_model('ai.h5')  # Replace with your model path

def predict(image):
    # Resize the image to match the model's expected input size (224x224)
    image = np.resize(image, (64, 64, 3))
    
    # Normalize the image (this is important for consistency with training)
    image = image / 255.0
    
    # Expand dimensions to add a batch dimension (expected input: [batch_size, height, width, channels])
    image = np.expand_dims(image, axis=0)
    
    
    # Make a prediction using the loaded Keras model
    predictions = model.predict(image)
    
    # Get the probabilities from the prediction result
    probabilities = predictions[0]
    
    # Define labels for your classes
    labels = {0: "healthy", 1: "leaf blight", 2: "leaf curl", 3: "non type", 4: "septoria leaf spot", 5: "verticulium wilt"}
    label_new = ["healthy", "leaf blight", "leaf curl", "non type", "septoria leaf spot", "verticulium wilt"]
    
    label_to_probabilities = []
    
    # Print the prediction probabilities
    print(probabilities)
    
    # Attach each label to its predicted probability
    
    # Sort the labels based on their probability score
    
    # Find the highest probability
    high = np.argmax(probabilities)
    result_1 = label_new[high]
    confidence = 100 * np.max(probabilities)
    
    # Prepare the result string with the category and confidence
    result = "Category: " + str(result_1) + "\n\n Confidence: " + str(confidence) + "%"
    
    return result

# Example usage with a PIL image
# image = Image.open('path_to_image.jpg')
# result = predict(image)
# print(result)


if __name__ == "__main__":
    main()
