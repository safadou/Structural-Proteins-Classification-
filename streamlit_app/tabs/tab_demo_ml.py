import numpy as np
import pandas as pd
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import preprocessing as pp

from config import models_dir


title = "Demo - Model ML"
sidebar_name = "Demo - Model ML"

#model_path = models_dir + "prot_clf_no_rfe_1.joblib"
model_path = models_dir + "ext_model.sav"
model_reduced_path= models_dir + "prot_clf_1.joblib"
prep_path = models_dir+ 'prot_preprocessing_1.joblib'

columns_clf = [ 'residueCount', 'resolution', 'structureMolecularWeight',
                'crystallizationTempK', 'densityMatthews', 'densityPercentSol',
                'macromoleculeType_DNA', 'macromoleculeType_Protein',
                'macromoleculeType_RNA', 'phValue_acide', 'phValue_basique',
                'phValue_neutre']

columns_rfe_clf = ['residueCount', 'resolution', 'structureMolecularWeight','crystallizationTempK', 'densityMatthews', 'densityPercentSol']

columns_ext_save = ['residueCount', 'resolution', 'structureMolecularWeight','crystallizationTempK', 'densityMatthews',
 'densityPercentSol', 'macromoleculeType_Protein', 'macromoleculeType_RNA']

#Input   : Dataframe to test
#Returns : Dataframe with prediction column, accuracy score
def ml_predict_with_dataframe(df):
    #Open the model file
    model = joblib.load(model_path)
    print("model opened")
    #make the prediction
    data = df
    #do this before providing the Dataframe
    data = data.dropna().reset_index(drop=True)
    y = 0
    if 'classification' in data.columns : 
        y = data.classification
        data = df.drop('classification', axis=1)

    to_drop = ['structureId', 'chainId', 'sequence', 'pdbxDetails', 'publicationYear', 'crystallizationMethod','experimentalTechnique']
    data = data.drop([x for x in to_drop if x in data.columns], axis=1)
    prep = pp.PreprocessingTransformer()
    data = prep.handle_missing(data)
    data = prep.reduce_modalities(data)
    data = prep.scale_encode_data(data)


    for col in columns_ext_save:
        if col not in data.columns:
            data[col] = 0

    data = data[columns_ext_save]

    print("Do the prediction: ")
    y_pred = model.predict(data)
    print("Prediction done")

    #insert predictions and give array back
    data['predicted_labels'] = y_pred
    data['true_labels'] = y

    return data

def ml_predict_with_user_input(input_dict):
    #Open the model with parameters reduced to 6
    model = joblib.load(model_path)
    
    prepro = joblib.load(prep_path)
    df = pd.DataFrame(input_dict, index=[0])
    #df_prep = prepro.scale_encode_data(df)
    
    df['macromoleculeType_Protein'] = 1
    df['macromoleculeType_RNA'] = 0
    #st.write(df_prep)
    #st.write(df)
    y_pred = model.predict(df)

    return y_pred


def run():

    st.title(title)
    
    with st.container():
        st.markdown("--------")
        st.subheader("Charger un fichier de test")

        with st.container():
            uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
            placeholder = st.empty()

            if uploaded_file is not None:
                print(uploaded_file)
                dataframe = pd.read_csv(uploaded_file)
                with placeholder.container():
                    with st.spinner("Please wait..."): 
                        df = ml_predict_with_dataframe(dataframe)
                    
                    st.dataframe(data=df[['true_labels', 'predicted_labels']]) 
            st.markdown("--------")

        with st.container():

            st.subheader("Saisie manuelle des paramètres")

            with st.form(key='user_inputs'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    residueCount = st.slider(
                                    'residuecount',
                                    6.0, 313236.0, step=1.0)

                    resolution = st.slider('resolution', 0.64, 50.0)
                    
                with col2:  
                    crystallizationTempK = st.slider('crystallizationTempK', 4.0, 398.0)

                    structureMolecularWeight = st.number_input('structureMolecularWeight')
                with col3:    
                    densityMatthews = st.slider('densityMatthews', 0.0, 99.0)
                    densityPercentSol = st.slider('densityPercentSol', 0.0, 92.0)
                
                submit_button  =  st.form_submit_button(label='Predict')
                            
                if submit_button:                       
                    input_dict={}
                    input_dict['residueCount']             = residueCount
                    input_dict['resolution']               = resolution
                    input_dict['structureMolecularWeight'] = structureMolecularWeight
                    input_dict['crystallizationTempK']     = crystallizationTempK
                    input_dict['densityMatthews']          = densityMatthews
                    input_dict['densityPercentSol']        = densityPercentSol


                    #Make prediction
                    placeholder2 = st.empty()
                    with st.spinner("Wait.."):
                        predicted_class = ml_predict_with_user_input(input_dict)
                        #print(predicted_class)
                        
                        with placeholder2.container() :
                            str = "Predicted class : "+ predicted_class[0]
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
                    

                    
                    
                    


   






