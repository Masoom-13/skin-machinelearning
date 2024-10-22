import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Create an uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the pre-trained Keras model
model = load_model('skin_model.h5')  # Ensure the .h5 file is in the same directory or provide the correct path

# Class labels (as defined in your training)
classes = {
    0: ('akiec', 'Actinic Keratoses and Intraepithelial Carcinomae'),
    1: ('bcc', 'Basal Cell Carcinoma'),
    2: ('bkl', 'Benign Keratosis-Like Lesions'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Melanocytic Nevi'),
    5: ('vasc', 'Pyogenic Granulomas and Hemorrhage'),
    6: ('mel', 'Melanoma'),
}


# Function to preprocess and predict using the provided PIL-based approach
def predict_image_pil(image_path):
    # Load and resize the image
    image = Image.open(image_path)
    image = image.resize((28, 28))

    # Convert the image to a numpy array and reshape it for model input
    img = np.array(image).reshape(-1, 28, 28, 3)

    # Make predictions using the loaded model
    result = model.predict(img)

    # Convert predictions to a list and find the class with the maximum probability
    result = result.tolist()
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)

    # Return the predicted class and the confidence score
    return class_ind, max_prob


# Streamlit app layout
st.title("Skin Cancer Detection App")
st.write("Upload an image to classify the type of skin lesion.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to the 'uploads/' folder
    file_path = os.path.join("uploads", uploaded_file.name)

    # Write the file to disk in the uploads folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Button to trigger prediction
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            # Make a prediction using the provided PIL code
            predicted_class, confidence = predict_image_pil(file_path)

            # Display the prediction result
            st.success(f"Prediction: {classes[predicted_class][1]}")
            st.write(f"Confidence: {confidence:.2f}")

            # Display detailed information about the predicted class
            st.write(f"Class ID: {predicted_class}")
            st.write(f"Class Name: {classes[predicted_class][0]}")

            # Print predictions and class details in the console as well
            print("Prediction probabilities for the uploaded image:")
            print(f"Predicted Class: {classes[predicted_class][1]}")
            print(f"Confidence: {confidence:.2f}")
else:
    st.warning("Please upload an image file to proceed.")
