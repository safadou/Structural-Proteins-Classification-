import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from config import data_dir, img_dir

title = "Données et méthodes"
sidebar_name = "Données et méthodes"

#data_path = 'C:/Users/engizulu/Documents/Projet_Datascientest/structural-protein-classification/data/'

description = st.container()
#img_dir = 'C:/Users/engizulu/Documents/Projet_Datascientest/structural-protein-classification/images/'

df_prot = pd.read_csv(data_dir+'data_no_dups.csv')
df_seq = pd.read_csv(data_dir+'data_seq.csv')
df = pd.merge(df_prot, df_seq, on=['structureId', 'macromoleculeType', 'residueCount'], how='inner')


def run():

    st.title(title)
    st.markdown("""Les données sont issues de la __[Protein Data Bank](https://www.rcsb.org/)__ (PDB)""")
    st.markdown("""Le dataset est constitué de deux fichiers :""")
    with st.container():
        st.markdown(""" 
                >> **data_no_dup.csv** (_proteins_):  
                regroupe toutes les variables décrivant les propriétés physiques des amino-acides (protéines)
                ainsi que les techniques utilisées pour les obtenir
                **(141401 lignes, 14 colonnes)**
                """)
        st.markdown(""" 
                |Features               |    Types         |         Description              |
                |:----------------------|-----------------:|------------------------------------------------:|
                | Classification        | object           | classification de la molécule                   |  
                | crystallizationMethod | object           | Méthode de cristallisation de la protéine       |
                | crystallizationTempK  | float64          | température à laquelle la protéine cristallise  |
                | densityMatthews       | float64          | volume cristallin par unité de poids moléculaire|
                | densityPercentSo      | float64          | % de la densité du solvant dans la protéine     |
                | experimentalTechnique | object           | Méthode d'obtention de  la structure protéique  |  
                | pdbxDetails           | object           | divers détails sur la protéine                  |
                | phValue               | object           | Ph de la solution (acide, basique,neutre)       |
                |publicationYear        | float64          | année de publication de la structure protéique  |
                |**residueCount**       | float64          | nombre d'acides aminés dans la séquence         |
                |resolution             | float64          | qualité du cristal contenant la protéine        |
                |structureMolecularWeight|float64          | masse moléculaire en kilo dalton                |
                |**structureId**        | float64          | id de la strcuture                              |
                |**macromoleculeType**  | object           | Type de macromolécule (Protein,DNA, RNA)        |   

                
                """)

        st.markdown(""" 
        

        """ )       
        st.markdown("""
                >> **data_seq.csv** (_séquences_):  
                contient les séquences associées à chaque acide aminé (protéine), ainsi que d'autres propriétés
                liées à la séquence
                **(467304 lignes, 5 colonnes)**
                """)
        st.markdown("""         
                |Features               |    Types         |    Description              |
                |:----------------------|-----------------:|------------------------------------------------:|          
                |**residueCount**       | float64          | nombre d'acides aminés dans la séquence         |
                |**macromoleculeType**  | object           | Type de macromolécule (Protein,DNA, RNA)        |
                |**structureId**        | float64          | id de la structure                              |
                |chainId                | float64          | id de la sequence                               |
                |sequence               | object           | sequence de la protéine                         |

            """) 
        st.markdown(""" 
        

        """ )    
        
        st.info(
            """
            Le merge des deux fichiers donne un dataset final de **471149 lignes, 16 colonnes** sur lequel portera l'étude.
            
            """)
    
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

            st.warning("""La variable cible ici est **classification**.\n """
                    """Elle présente **4989** modalités différentes!!""")
        with st.container():
            st.subheader("Distribution des classes les plus présentes dans le dataset")
            st.image(img_dir+'pie_classes.png')

            st.success("Nous avons présenté uniquement les 10 classes les plus répandues dans le dataset. "
                    "La classe **RIBOSOME** est la plus répandue et représente plus de **17%**  du total des classes du"
                    "dataset complet et 24% lorsqu'on ne considère que les 10 plus classes les plus importantes")

        with st.container():
            st.markdown(
                """

                """
            )





