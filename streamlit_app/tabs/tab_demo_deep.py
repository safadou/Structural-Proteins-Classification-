from logging import PlaceHolder
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import label_binarize

import streamlit as st
import pandas as pd
import numpy as np
#import preprocessing as pp

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence, text
from keras.preprocessing.text import Tokenizer

from config import models_dir


title = "Demo - Model DL"
sidebar_name = "Demo - Model DL"

model_path = "../models/lstm/model_deep_lstm.pkl"
tokenizer_path = "../models/lstm/tokenizer_deep.sav"
labelbinarizer_path = "../models/lstm/label_binarizer.sav"

def load_models():
    #CNN Model
    cnn_model =  tf.keras.models.load_model(model_path)
    
    #Tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    #Label Binarizer
    with open(labelbinarizer_path, 'rb') as handle:
        label_binarizer = pickle.load(handle)
    
    return cnn_model, tokenizer, label_binarizer

def dl_cnn_predict(seq, model, tokenizer, lb, maxlen=268):
    
    #tokenize sequence
    X = tokenizer.texts_to_sequences(seq)
    X = sequence.pad_sequences(X, maxlen=maxlen)

    #Make prediction
    y_pred = model.predict(X)
    #Inverse transform label
    y = lb.inverse_transform(y_pred)
    return y

def run():

    st.title(title)

    #open model
    model, tokenizer, lb = load_models()

    with st.container():
        st.subheader("Test DL Model")
        st.markdown("--------")


    with st.container():
        uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
        placeholder = st.empty()

        if uploaded_file is not None:
            print(uploaded_file)
            dataframe = pd.read_csv(uploaded_file)
            with placeholder.container():
                with st.spinner("Please wait..."): 
                    y_pred = dl_cnn_predict(dataframe.sequence, model, tokenizer, lb) 
                    with placeholder.container():
                        dataframe['predicted_labels']= y_pred
                        st.write("Prediction class : ")
                        st.write(dataframe[['classification', 'predicted_labels']])
                            
                
                #st.dataframe(data=df[['true_labels', 'predicted_labels']]) 
        st.markdown("--------")


    with st.form(key='user_deep_inputs'):
        #col1, col2 = st.columns(2)
        #with col1:
        with st.container():
            sequence  = st.text_input('Sequence')
            submit_button  =  st.form_submit_button(label='Predict')
                        
        #with col2:  
            placeholder = st.empty()
            if submit_button:
                with placeholder.container():                     
                    #Make prediction
                    with st.spinner("Wait.."):
                        y_pred = dl_cnn_predict([sequence], model, tokenizer, lb) 
                        #with placeholder.container():
                        #st.write("Predicted class : "+y_pred[0])
                        #st.write(y_pred)
                        str = "Predicted class : "+ y_pred[0]
                        html_str = f"""
                                    <style>
                                    p.a {{
                                    font: bold 24px Courier;
                                    color: red;
                                    }}
                                    </style>
                                    <p class="a">{str}</p>
                                    """
                        st.markdown(html_str, unsafe_allow_html=True)
                    
                            
                    


   






