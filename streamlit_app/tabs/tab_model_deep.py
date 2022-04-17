import streamlit as st
#import pandas as pd
#import numpy as np
#from PIL import Image

from config import img_dir

title = "Analyse de la séquence - Model DL"
sidebar_name = "Model DL"
#img_dir = '../images/'

def run():

    st.title(title)

    st.info(
        """
        **Le but poursuivi est d'arriver à prédire la variable **classification** à partir de la séquence**
        """
    )

    with st.container():
        st.markdown(
            """
            Une protéine est une macromolécule constituée par l'enchaînement de molécules de masse moléculaires plus petites
            appelées monomères ou acides aminés. Ils sont représentés par des lettres [A...Z]
            """
        )
        st.image(img_dir+'sequence_example.png')
        #st.write(pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD")))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                Longueurs des chaines de séquences dans le dataset
                """
            )
            st.image(img_dir+'dist_seq_len.png')

        with col2:
            st.markdown(
                """
                Fréquence des lettres dans les séquences

                
                """
            )
            st.image(img_dir+'seq_codes_frequency.png')

        st.subheader("Preprocessing")

        st.write(
            """
            ### Tokenization
            - Problème de sequence-to-sequence learning 
            - Tokenization classique, avec modèle réduit aux chaines de longueur < 268 caractères et padding des séquences inférieures
            (Initialement testé avec la longueur maximale, puis avec une longueur de 1000 caractères)
            """
        )

        st.write(
            """
            ### Algorithmes testés
            1. CNN 1D

            """
        )
        st.image(img_dir+'summary_sequential_cnn1_deep.png')

        st.write(
            """
            2. LSTM
            
            """
        )


        