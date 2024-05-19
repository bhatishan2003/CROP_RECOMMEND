# import streamlit as st
# import pickle
# import numpy as np
# #st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")
# # Load the model

# with open('crop_recommendation_model2.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Display the HTML content in the sidebar
# # Function to make predictions
# def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
#     prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
#     return prediction[0]
# # Streamlit UI
# st.title('Crop Recommendation System')
# st.sidebar.title('Parameters')

# N = st.number_input('Nitrogen (N)', min_value=0, max_value=200)
# P = st.number_input('Phosphorous (P)', min_value=0, max_value=100 )
# K = st.number_input('Potassium (K)', min_value=0, max_value=250)
# temperature = st.number_input('Temperature (°C)', min_value=0, max_value=50)
# humidity = st.number_input('Humidity (%)', min_value=0, max_value=100)
# ph = st.number_input('pH', min_value=0, max_value=14)
# rainfall = st.number_input('Rainfall (mm)', min_value=0, max_value=1000)
# # st.subheader('Enter the parameters on the left sidebar and click Predict')

# if st.button('Predict'):
#     result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
#     st.success(f'The recommended crop is {result}')

import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from PIL import Image

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='wide', initial_sidebar_state="collapsed")
def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    # Title
    html_temp = """
    <div style="background-color:#7ad961;padding:10px;border-radius:10px">
    <h1 style="color:green;text-align:left;"> Crop Recommendation System 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #Background image
    #image_path = "C:/Users/DELL/OneDrive/Desktop/CROP/istockphoto-1153409397-1024x1024.jpg"
    #image = Image.open(image_path)
    #st.image(image, width=400, caption='Farmer planting sprout in soil')

    col1, col2 = st.columns([2, 2])
    with col1: 
        with st.expander(" ℹ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)
    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosphorus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):
            loaded_model = load_model('crop_recommendation_model3.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
            ## Results 🔍 
            ''')
            col1.success(f"{prediction.item().title()} is recommended for your farm.")

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()