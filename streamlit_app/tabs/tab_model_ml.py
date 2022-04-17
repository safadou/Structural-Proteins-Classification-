from nbformat import write
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import joblib
import preprocessing as pp

from config import img_dir

title = "Analyse des propriétés physiques - Model ML"
sidebar_name = "Model ML"
#img_dir = '../images/'


def run():

    st.title(title)

    st.info(
        """
        Le but poursuivi est d'arriver à prédire la variable **classification** à partir des élements autre que la séquence
        """
    )

    st.markdown(
        """
        Les étapes de l'analyse :

        1. Préprocessing
        2. Feature reduction & conclusions sur les variables
        3. Sélection des modèles ML et analyse des résultats
        4. Démo
        
        """
    )

    st.subheader("1. Preprocessing")
    #missing_img = img_dir+'missing_values.png'
    st.markdown(
    """
    Préalable à cette opération : Une étude approfondie de la définition de chaque variable, son type,
    son impact sur la variable cible (correlation), ainsi que la proportion de valeurs nulles associées.
    On a déduit les étapes suivantes pour préparer les données :

    **1. Suppression des colonnes inutiles, ou à caractères uniquement informatives**
    >> 
    >> - 'sequence' : L'analyse se fait ici sans cette variable expréssement
    >> - 'structureId' : sert uniquement d'index 
    >> - 'chainId' : sert uniquement d'index
    >> - 'pdbxDetails' : contient diverses informations sur la molécule. Ne peut être exploitée dans l'état
    >> - 'publicationYear' : année de publication de l'étude sur la molécule. N'a aucun impact sur sa classification

    **2. Gestion des valeurs nulles (NaN)**
    >> Le dataset présente beaucoup de valeurs nulles. Supprimer directement revient à perdre énormément de données. Une analyse 
    >> détaillée a été nécessaire. Les valeurs numériques ont été en majorité remplacée par la médiane, et les valeurs catégorielles par le mode
    
    """)
    st.image(img_dir+'missing_values.png')
    st.markdown(""" 

    **3. Reduction des modalités des variables**

    Il s'est agit d'analyser chacune des variables catégorielles pour ne conserver que les modalités importantes et éviter un nombre trop important de
    de variables lors de l'encoding en variables numériques.
    Les modalités peu representées ont parfois été regroupées en une seule, pour éviter de perdre trop de données en les supprimant.

    """
    )
    st.markdown(
        """
        **4. Vérification et correction de l'assymétrie des données**

        Remarquée lors de l'analyse exploratoire, certaines variables présentent une asymétrie (skewness), que nous avons corrigé
        notamment en remplaçant la valeur de la variable en prenant son logarithme. Cela a permit de corriger cet effet et assurer une
        distribution équilibrée.

        **5. Scaling et encodage**
        
        Le scaling des variables numériques et l'encodage des variables catégorielles en variables numériques. Nous avons testé les scalers
        RobustScaler et StandardScaler. Le RobustScaler est moins sensible aux outliers, et notre dataset présente plusieurs valeurs extrêmes
        dont on ne sait déterminer si elles sont abbérantes ou pas, partant du fait qu'il s'agit de mesures physiques pour la plupart.
        """
    )

    st.subheader("2. Features reduction")

    st.markdown(
        """
        Le dataset complet après encoding des variables catégorielles compte plus de 26 variables. 
        Nous avons procédé par comparaisons de plusieurs algorithmes pour dégager le set minimum de variables permettant d'obtenir
        des performances maximales et une accuracy satisfaisante.
        Méthodes testées :

        1. RandomForest

        """
    )
    st.image(img_dir+'importance_random_forest.png')

    st.markdown(
        """
        2. SelectFromModel avec DecisionTreeClassifier()
        """
    )
    st.image(img_dir+'importance_decisiontree.png')

    st.markdown(
        """
        3.Logistic Regression
        """
    )
    st.image(img_dir+'importance_logistic_reg.png')
    st.markdown(
        """
        Correspondant aux features suivantes retenues : 
    """)
    st.image(img_dir+'importance_logistic_reg_list.png')

    st.markdown("Nous avons testé également d'autres modèles tels")

    st.markdown(
        """
        4.RFE

        Le modèle RFE a été testé avec plusieurs classifieurs différents (ExtraTreesClassifier, LogisticRegression), et de manière globale, il nous
        a permis de garder comme variables importantes les variables suivantes :  
        """
    )
    st.image(img_dir+'importance_final_values.png')

    st.info(
        """
        Le modèle final prendra donc en entrée uniquement **8 variables** sur la vingtaine disponible avant réduction et permet d'obtenir de
        très bonnes performances malgré cela.
        """)

    st.subheader("3. Sélection des modèles ML")

    st.info(
    """
    Le problème à résoudre est un problème de classification multinomial avec 16 classes au total. 
    """
    )
    st.markdown(
    """
    Le dataset présentait initialement beaucoup de classes dans la variable cible. Nous avons réduit ces classes à celles regroupant 
    plus de 5000 observations. Cela fait un total de 16 classes.
    Ce choix a été assumé pour éviter une trop grande réduction de la taille des données disponibles, et aussi une distribution non équilibrée des données cibles.

    Pour trouver le modèle présentant les meilleures performances, notamment un taux de prédiction élevé, nous avons testé plusieurs modèles
    de classification : 
    """
    )

    st.image(img_dir+'model_accuracy.png')

    st.markdown(
    """
    11 modèles ont été testés, puis l'étude a été restreinte aux 4 modèles présentant les meilleures performances.
    Nous avons étudié l'évolution des courbes d'apprentissage pour déterminer si nos modèles faisaient du surapprentissage,
    et également recherché les hyperparamètres qui optimisaient le mieux le modèle retenu parmi ces 4.   
    """)

    st.markdown("""Learning Curves :""")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_dir+'learning_curves_extratrees.png')
        with col2:
            st.image(img_dir+'learning_curves_random_forest.png')
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_dir+'learning_curve_decision_tree.png')
        with col2:
            st.image(img_dir+'learning_curves_gradient_boosting.png')
    

    st.markdown(
    """
    ##### Matrices de confusions réduites

    Pour illustrer la performance des modèles choisis, nous présentons ici des matrices de confusions des deux meilleurs modèles
    réduites aux 4 premières classes : 
    """)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_dir+'matrice_confusion_extratrees.png')
        with col2:
            st.image(img_dir+'matrice_confusion_random_forest.png')


    
    




        