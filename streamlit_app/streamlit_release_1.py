# -*- coding: utf-8 -*-

"""
Created on Sat Oct 30 21:22:48 2021

@author: Utilisateur DIALLO Sadou Safa & NGIZULU Edi

"""
# Calculation
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as stc

IMG_DIR = '../images/'

im = Image.open(IMG_DIR + "icon.png")
# st.set_page_config(
#     page_title='Protein Classification Project',
#     page_icon=im,
#     layout="centered",  # wide,
#     initial_sidebar_state="auto")

img = Image.open(IMG_DIR + 'sequence_p.png')

HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;"> Structural Protein Classification </h1>
    </div>
 """

header = st.container()
description = st.container()

# Chargement des fichiers de data
df_prot = pd.read_csv('../data/data_no_dups.csv')
df_seq = pd.read_csv('../data/data_seq.csv')
df = pd.merge(df_prot, df_seq, on=['structureId', 'macromoleculeType', 'residueCount'], how='inner')


def presentation():
    with header:
        st.image(img)
        stc.html(HTML_BANNER)
        # st.subheader("By __DIALLO__ Sadou Safa & __NGIZULU__ Edi")

        st.subheader('__Présentation du Projet__')

        st.markdown("""__Les protéines__ sont des macromolécules complexes, elles sont les plus abondantes des 
        molécules organiques des cellules et constituent à elles seules plus de **50%** du poids\n à sec des êtres 
        vivants. <br> La protein data bank (**PDB**) est une base de donnée mondiale qui recense les données sur le 
        séquençage  des protéines, les techniques utilisées pour les obtenir ainsi que les classifications et autres détails
        obtenus suite à l'analyse de la molécule.<br> Pour des amples informations, vous pouvez visiter leur site [ici](https://www.rcsb.org/). <br> 
        La PDB est une base de données importante dans la recherche médicale car en plus de permettre des avancées 
        dans la connaissance de la composition des protéines et de leur utilité, elle permet aussi de faire des tests 
        sur de nouvelles molécules destinées à lutter contre les maladies émergentes comme le SARS-COV2. <br> """,
                    unsafe_allow_html=True)
        st.subheader("__Quels objectifs pour ce projet ?__")
        st.markdown(""" __L'objectif principal__ de ce projet est la classification par des _algorithmes d'ensemble_ de la structure des protéines \
                après séquençage en analysant les variables descriptives fournies par le PDB à l'exception de la variable séquence. <br>
                En effet, il est possible d'obtenir la classification d'une protéine en étudiant uniquement sa séquence, qui décrit la liste \
                d'acides aminés composant cette molécule. Cette méthode a notamment été largement abordée notamment sur Kaggle.
                    __L'objectif secondaire__ est la description analytique de toutes les variables des datasets fusionnés.
                """, unsafe_allow_html=True)

        st.subheader("__Rappel du Contexte__")
        st.markdown("""Après un premier projet recusé en raison de la volumétrie des données  jugée insuffisante, \
                    la classification de la strcuture des proteines fut quant à lui en terme volumétrique suffisamment dense \
                    pour que des puissants algorithmes de classification n'en viennent à bout.\
                    La prédiction de la structure des proteines est une discipline majeure en bio-informatique et en chimie, \
                    elle trouve ses applications dans l'industrie pharmaceutique (fabrication des nouveaux médicaments), \
                    en bio-technologie (conception de nouvelles enzymes) ou encore dans de nouvelles start-ups orientées vers la recherche appliquée sur les proteines.
                    Nous avons analysé deux datastes extraits de la banque mondiale de proteine, datastes composé pour l'un par les proteines et pour l'autre\
                    les séquences associées à ces proteines. <br>
                    __A la différence de beaucoup de compétiteurs dans [Kaggle](https://www.kaggle.com/shahir/protein-data-set), \
                    nous avons fait fait le choix d'utiliser un minimum des variables explicatives  du fichier de fusion après réduction de dimension. <br>
                    Notre crainte d'avoir un modèle fortement dependant de la variable séquence, nous a fait écarter celle-ci.__
                    """, unsafe_allow_html=True)
        st.info("""Dans la barre de gauche nos analyses détaillées et interactives vous permettront de naviguer plus facilement.\
         Vous avez aussi la possibilité de fournir à l'algorithme au choix les données des variables pour obtenir une prédiction de la classe de proteine\
             correspondante.""")

    return


# def color_column(s, df1, df2):
#    return ['background-color: green'] * len(s) if s in df1 and s in df2 else ['background-color: red'] * len(s)


def dataset():
    with description:
        st.subheader("Le dataset est constitué de deux fichiers : ")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                # container = st.container()
                st.subheader('data_no_dup.csv')
                """Contient les propriétés physiques des macromolécules étudiées"""
                st.write("""__{} lines, {} columns__""".format(df_prot.shape[0], df_prot.shape[1]))
                df_tmp = pd.DataFrame(df_prot.columns, columns=['Features'])
                st.dataframe(df_tmp.style.hide_index())
            with col2:
                # container = st.container()
                st.subheader('data_seq.csv')
                """Contient les informations sur le séquençage de chaque protéine"""
                st.write('__{} lines, {} columns__'.format(df_seq.shape[0], df_seq.shape[1]))
                df_col = pd.DataFrame(df_seq.columns, columns=['Features'])
                st.write(df_col.style.hide_index())
                # if st.checkbox('Aperçu des données'):
                #    st.dataframe(df_prot.head(50))

        st.info(
            """Le merge des deux fichiers donne un dataset final de **{} lignes, {} colonnes** sur lequel portera l'étude.""".format(
                df.shape[0], df.shape[1]))
        # with st.expander("Données manquantes "):
        #    df_nan = pd.DataFrame(
        #        round((df.isnull().sum().sort_values(ascending=False) / df.shape[0]) * 100, 1)).reset_index()
        #    df_nan.columns = ['Columns', '% of missing values']
        #    df_nan

        #    st.info(
        #        "**Note:** La variable cible (**classification**) n'a aucune donnée manquante (4989 classes de proteines!), de même que la séquence associée à la classification ")

        with st.container():
            st.warning(
                "Il est important de comprendre la signification de chaque variable fournie dans le dataset")
            df_feat = pd.read_csv('../data/features_description.csv', sep=';', index_col=0)

            st.table(df_feat.dropna())
        # st.subheader("__Analyse des Résultats__")


def missing_value_table(df):
    missing_value = df.isnull().sum().sort_values(ascending=False)
    missing_value_percent = round(missing_value * 100 / df.shape[0], 1)
    missing_value_t = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_t.rename(columns={0: 'Missing ',
                                                                 1: '% '})
    cm = sns.light_palette('red', as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return


def eda():
    # Display an extract of the dataset here on demand
    st.subheader("Aperçu rapide du dataset ")
    with st.container():
        if st.checkbox("Afficher un extrait"):
            st.dataframe(df.head())

    with st.container():
        if st.checkbox("Synthèse variables numériques"):
            st.dataframe(df.describe().transpose())

    with st.container():
        if st.checkbox("Synthèse variables catégorielles"):
            st.dataframe(df.describe(include='object').transpose())

        st.warning("""La variable cible ici est la **classification**.\n """
                   """Elle présente **4989** modalités différentes""")
    with st.container():
        st.subheader("Distribution des classes les plus présentes dans le dataset")
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(aspect="equal"))
        data = df.classification.value_counts().sort_values(ascending=False)[:10]
        wedges, texts, autotexts = ax.pie(data, wedgeprops=dict(width=0.5), autopct='%1.1f%%')
        ax.legend(wedges, data.index, title="classification", bbox_to_anchor=(1, 0, 0.5, 1), loc="center left")
        plt.setp(autotexts, size=12, weight="bold")
        st.pyplot(fig)

        st.info("Nous avons présenté uniquement les 10 classes les plus répandues dans le dataset. "
                "La classe **RIBOSOME** est la plus répandue et représente plus de **17%**  du total des classes du"
                "dataset complet et 24% lorsqu'on ne considère que les 10 plus classes les plus importantes")

    with st.container():
        st.subheader("Nettoyage du dataset")
        with st.expander("Données manquantes"):
            st.write(missing_value_table(df))

    # with nums:
    #    with st.expander("Description des variables numériques"):
    #        st.write(df.describe(include='number'))
    #        st.info("**Remarque:** l'année de publication maximale sur la classification des proteines reste 2018 , "
    #                "nous constatons aussi une distribution non homogène des données de la variable residueCount")
    # Merging

    # with categs:
    # Bug streamlit : describe() échoue pour les variables catégorielles / trouver une autre manière d'afficher ça
    #    with st.expander("Description des variables catégorielles"):
    # st.write(df.describe(exclude='number'))
    #        st.info("**Remarque:** 60710 occurences de classifications des protéines réparties en 4989 classes (type "
    #                "de classification), "
    #                "la classe **RIBOSOME** est plus répandue et représente plus de **17%** de type de "
    #                "classification.")
    #        #    st.write(df['classification'].describe())
    return


def feat_selection():
    #with st.container():

    return


def visualisation():
    with header:
        st.title('Visualisations')

    with st.container():
        with st.expander("Analyse de la variable macromoleculeType"):
            # st.bar_chart(df.macromoleculeType.value_counts().sort_values(ascending=False), use_container_width=True)
            fig = plt.figure(figsize=(12, 8))
            df.macromoleculeType.value_counts().sort_values().plot(kind='barh', fig=fig)
            st.pyplot(fig)

            fig = plt.figure(figsize=(12, 8))
            # all_columns = df.columns.to_list()

            pieplot = df.macromoleculeType.value_counts().sort_values(ascending=False)[:5].plot.pie(autopct="%1.1f%%",
                                                                                                    title="Répartition de la variable macromolécule selon le type")
            # fig = plt.pie(x = range(len(df.macromoleculeType),labels = df.macromoleculeType.unique(), autopct="%1.1%%", fig=fig)
            st.write(pieplot)
            st.pyplot(fig)

            st.info("**75%** des macromolécules type sont des proteines, **24%** sont un melange entre proteines, "
                    "DNA et RNA, le tout représentant **96%** des macromolécules Restriction du jeu de données aux "
                    "macromolécules les plus présentes. Le jeu de données a été restreint à ces 3 catégories de macromolécules.")

        with st.expander("Analyse de la variable classification"):
            # st.bar_chart(df.classification.value_counts().sort_values(ascending=True)[:50])
            st.subheader("Diagramme des 50 classes les plus présentes dans le dataset")
            fig = plt.figure()
            df.classification.value_counts().sort_values(ascending=False)[:50].plot(kind='bar', figsize=(12, 8))
            st.pyplot(fig)

            # fig = plt.figure(figsize = (12,8))
            # all_columns = df.columns.to_list()

            # cplot = df.classification.value_counts().sort_values(ascending=False)[:10].plot.pie(autopct="%1.1f%%",title="Répartition de la variable classification")
            # fig = plt.pie(x = range(len(df.macromoleculeType),labels = df.macromoleculeType.unique(), autopct="%1.1%%", fig=fig)
            # st.write(cplot)
            # st.pyplot(fig)

            # Pie plot classification
            st.markdown("### Les 4 classes les plus présentes dans le dataset")
            img = Image.open('dataset/figure/target_donut.png')
            st.image(img)
            st.info(
                " Les 4 classes les plus présentes sont le Ribosome, l'hydrolase, le transferase et l'oxydoreductase sur un totale de : ")
            st.write(df.classification.nunique())
        with st.expander("Analyse de la variable Méthode de crystallization  "):
            st.markdown("### Barplot des 10 méthodes de crystallization ")

            fig = plt.figure()
            df.crystallizationMethod.value_counts()[:10].sort_values(ascending=False).plot(kind="barh")
            st.pyplot(fig)

            # df.crystallizationMethod.value_counts().sort_values(ascending=False)[:5].plot(kind = 'pie', ax=ax, title="Les 5 plus grandes méthodes de crystallisation dans le dataset")
            fig = plt.figure(figsize=(12, 8))
            # all_columns = df.columns.to_list()
            st.markdown("### Les 10 méthodes de cristallisation les plus présentes ")
            pieplot_c = df.crystallizationMethod.value_counts().sort_values(ascending=False)[:5].plot.pie(
                autopct="%1.1f%%")
            # fig = plt.pie(x = range(len(df.macromoleculeType),labels = df.macromoleculeType.unique(), autopct="%1.1%%", fig=fig)
            st.write(pieplot_c)
            st.pyplot(fig)
            st.info(
                "Les 3 méthodes de cristallisation les plus répresentatives font plus de 95% et ont été retenues dans le modèle final  ")

            return


def modelisation():
    with st.container():
        st.header("Modelisation")
        df2 = df.dropna().drop(['sequence', 'structureId', 'pdbxDetails', 'publicationYear', 'chainId'], axis=1)
        st.info('New Size : __{} lines, {} columns__'.format(df2.shape[0], df2.shape[1]))
    return


def interpretation():
    st.write('interpretation')
    return


if __name__ == "__main__":
    side_actions = ['Présentation du Projet', 'Datasets', 'Analyse exploratoire', 'Selection de variables',
                    'Visualisation', 'Modélisation',
                    'Conclusion']
    # st.sidebar.image('protein-structure-title-image.jpeg', use_column_width=True)

    st.sidebar.image(IMG_DIR+'proteins.png', use_column_width=True)
    action = st.sidebar.radio("", side_actions, index=0)
    actions_functions = [presentation, dataset, eda, feat_selection, visualisation, modelisation, interpretation]
    actions_functions[side_actions.index(action)]()
# Title of the projet
# Header that contains description, and purpose of the project


st.sidebar.info("""Auteurs:\n
 __DIALLO__ Sadou Safa  [linkedin](https://www.linkedin.com/in/sadou-safa-diallo-a0839b49/)\n
 __NGIZULU__ Edi   [linkedin](www.linkedin.com/in/edi-ngizulu-57256316a)\n
 Formation continue Data Scientist Mai 2021, [Data Scientest](https://datascientest.com/)\n
 Source de données: [kaggle](https://www.kaggle.com/shahir/protein-data-set)""")
