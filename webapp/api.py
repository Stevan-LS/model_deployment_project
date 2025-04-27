import streamlit as st
import random as rd
import base64
import requests

API_URL = "http://serving-api:8080/predict"
REPORTING_URL = "http://serving-api:8080/feedback"
cifar10_classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]



def get_prediction(image_file):
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
       
    payload = {"image": encoded_image}
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}

def send_feedback(image_file, prediction, feedback):
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    payload = {
        "image": encoded_image, 
        "prediction": prediction, 
        "feedback": feedback
    }
    response = requests.post(REPORTING_URL, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}

# Initialize session state for prediction and uploaded image
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "incorrect_prediction_clicked" not in st.session_state:
    st.session_state.incorrect_prediction_clicked = None

# To access file uploader instance, use st.session_state[st.session_state.file_uploader_key]

st.title("Title")
st.header("Predict")

# Function to clear the prediction when a new image is uploaded
def new_image():
    st.session_state.prediction = None

# Add a file uploader for images, tied to session state
st.file_uploader(
    label="Upload an image to predict the class (png, jpg, jpeg).",
    type=["png", "jpg", "jpeg"],
    key=st.session_state.file_uploader_key,
    on_change=new_image,
)


    
# Store the uploaded file in session state
if st.session_state[st.session_state.file_uploader_key] is not None:
    st.image(st.session_state[st.session_state.file_uploader_key])

# Display the Predict button only if an image is uploaded
if st.session_state[st.session_state.file_uploader_key] and st.button("Predict"):
    st.session_state.prediction = get_prediction(st.session_state[st.session_state.file_uploader_key]) # CALL API PREDICT

# Display the prediction result
if st.session_state.prediction is not None:
    article = "an" if st.session_state.prediction["prediction"][0] in "AEIOU" else "a"
    entity = st.session_state.prediction['prediction']
    if entity is not None:
        st.write(f"It's {article} {entity}!")

    # Correct Prediction
    if st.button("Correct Prediction"):
        # CALL API FEEDBACK
        send_feedback(st.session_state[st.session_state.file_uploader_key], st.session_state.prediction["prediction"], st.session_state.prediction["prediction"])
        print(f"Call API:\nimage : {st.session_state[st.session_state.file_uploader_key]}\nprediction : {st.session_state.prediction}\ntruth value : {st.session_state.prediction}")
        
        # Clear the session state and reset the app
        st.session_state.prediction = None
        st.session_state.incorrect_prediction_clicked = None
        st.session_state.file_uploader_key += 1
        st.rerun()
    
    # Incorrect Prediction
    if st.button("Incorrect Prediction"):
        st.session_state.incorrect_prediction_clicked = True
    
    # Correct the inccorrect prediction
    if st.session_state.incorrect_prediction_clicked:
        feedback = st.selectbox("Select the correct class", cifar10_classes)
        
        if st.button("Submit Report"):
            # CALL API FEEDBACK
            send_feedback(st.session_state[st.session_state.file_uploader_key], st.session_state.prediction["prediction"], feedback)
            print(f"Call API:\nimage : {st.session_state[st.session_state.file_uploader_key]}\nprediction : {st.session_state.prediction}\ntruth value : {feedback}")
            
            # Clear the session state and reset the app
            st.session_state.prediction = None
            st.session_state.incorrect_prediction_clicked = None
            st.session_state.file_uploader_key += 1 
            st.rerun()

# Debugging
#st.text(f"uploder {st.session_state[st.session_state.file_uploader_key]}\n prediction {st.session_state.prediction} \n key {st.session_state.file_uploader_key}")