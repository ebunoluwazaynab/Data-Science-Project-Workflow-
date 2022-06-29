import streamlit as st
import numpy as np
import pickle

def load_model():
    with open('model_building_steps (3).pkl', "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()
model = data['model']
le = data['le'] 
scaler = data['scaler']

def prediction_page():
    st.title("Iris Flower Specie Prediction")
    st.write("""This app predicts the specie of an Iris flower""")
    
    
    st.write("""#### Input Parameters""")
    
    Sepal_Length = st.slider('Sepal Length', 8.0, 5.0)
    Sepal_Width = st.slider('Sepal Width', 5.0, 2.0)
    Petal_Length = st.slider('Petal Length', 7.0, 1.0)
    Petal_Width = st.slider('Petal Width', 3.0, 0.0)
    
    
    ok = st.button('Check Flower Specie')
    
    if ok:
        Features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
        Features = scaler.transform(Features)
        Prediction = model.predict(Features)
        
        st.subheader('The species of the iris flower is' + ' ' + le.classes_[Prediction[0]])

        
  
        
    
    

    
    
    
    
    
