###########
# Imports #
###########

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

im = Image.open("icon.png")
# st.set_page_config(
#     page_title='Protein Classification Project',
#     page_icon=im,
#     layout="centered",  # wide,
#     initial_sidebar_state="auto")

header = st.container()
description = st.container()

# Chargement des fichiers de data
df_prot = pd.read_csv('dataset/data_no_dup.csv')
df_seq = pd.read_csv('dataset/data_seq.csv')
df = pd.merge(df_prot, df_seq, on=['structureId', 'macromoleculeType', 'residueCount'], how='inner')


def introduction():
    with header:
        st.title("Classification de la structure des protéines")
        st.header('Objectif du projet')
        "Prédiction de la classification protéinique après séquençage, basée sur les caractéristiques physiques " \
        "fournies dans le dataset (autres que la séquence). "

    with description:
        st.header("Présentation du Dataset")
        with st.expander("Merge des données : "):
            st.subheader("Le dataset est constitué de deux fichiers :")
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    # container = st.container()
                    st.subheader('data_no_dup.csv')
                    """Contient les propriétés physiques des macromolécules étudiées"""
                    st.markdown("""**{} lines, {} columns**""".format(df_prot.shape[0], df_prot.shape[1]))
                    df_tmp = pd.DataFrame(df_prot.columns, columns=['Features'])
                    st.dataframe(df_tmp.style.hide_index())
                with col2:
                    # container = st.container()
                    st.subheader('data_seq.csv')
                    """Contient les informations sur le séquençage de chaque protéine"""
                    st.write('{} lines, {} columns'.format(df_seq.shape[0], df_seq.shape[1]))
                    df_col = pd.DataFrame(df_seq.columns, columns=['Features'])
                    st.write(df_col.style.hide_index())
                    # if st.checkbox('Aperçu des données'):
                    #    st.dataframe(df_prot.head(50))

            st.info("""On fait un merge des deux fichiers :  
                       - Dimensions du dataset après merge : **{} lignes, {} colonnes**""".format(df.shape[0],
                                                                                                  df.shape[1]))

        with st.expander("Données manquantes :"):
            df_nan = pd.DataFrame(
                round((df.isnull().sum().sort_values(ascending=False) / df.shape[0]) * 100, 1)).reset_index()
            df_nan.columns = ['Columns', '% of missing values']
            df_nan

            st.info(
                "**Note:** La variable cible (**classification**) n'a aucune donnée manquante, de même que la séquence associée à la classification !")

        with st.expander("Details des variables (glossaire) : "):
            df_feat = pd.read_csv('features_description.csv', sep=';')
            df_feat
    return


def color_column(s, df1, df2):
    return ['background-color: green'] * len(s) if s in df1 and s in df2 else ['background-color: red'] * len(s)


def eda():
    nums = st.container()
    categs = st.container()

    with header:
        st.title('Exploratory Data Analysis')

    with nums:
        with st.expander("Description des variables numériques"):
            st.write(df.describe())
            st.info("**Remarque:** l'année de publication maximale sur la classification des proteines reste 2018 , "
                    "nous constatons aussi une distribution non homogène des données de la variable residueCount")
        # Merging

    with categs:
        # Bug streamlit : describe() échoue pour les variables catégorielles / trouver une autre manière d'afficher ça
        with st.expander("Description des variables catégorielles"):
            st.info("**Remarque:** 60710 occurences de classifications des protéines réparties en 4989 classes (type "
                    "de classification), "
                    "la classe **RIBOSOME** est plus répandue et représente plus de **17%** de type de "
                    "classification.")
            #    st.write(df['classification'].describe())
    return


def visualisation():
    with header:
        st.title('Visualisations')

    with st.container():
        with st.expander("Analyse de la variable macromoleculeType"):
            # st.bar_chart(df.macromoleculeType.value_counts().sort_values(ascending=False), use_container_width=True)
            fig = plt.figure()
            df.macromoleculeType.value_counts().sort_values().plot(kind='barh', fig=fig)
            st.pyplot(fig)

            st.info("**75%** des macromolécules type sont des proteines, **24%** sont un melange entre proteines, "
                    "DNA et RNA, le tout répresnetant **96%** des macromolécules Restriction du jeu de données aux "
                    "macromolécules les plus présentes")

        with st.expander("Analyse de la variable classification"):
            #st.bar_chart(df.macromoleculeType.value_counts().sort_values(ascending=True))
            fig = plt.figure()
            df.classification.value_counts().sort_values(ascending=False)[:50].plot(kind='bar', fig=fig)
            st.pyplot(fig)


    return


def modelisation():
    st.write('modélisation')
    return


def interpretation():
    st.write('interpretation')
    return


if __name__ == "__main__":
    side_actions = ['Introduction', 'Analyse exploratoire', 'Visualisation', 'Modélisation', 'Interprétation']
    # st.sidebar.image('protein-structure-title-image.jpeg', use_column_width=True)
    st.sidebar.image('proteins.jpeg', use_column_width=True)
    action = st.sidebar.radio("", side_actions, index=0)

    actions_functions = [introduction, eda, visualisation, modelisation, interpretation]
    actions_functions[side_actions.index(action)]()
# Title of the projet
# Header that contains description, and purpose of the project
