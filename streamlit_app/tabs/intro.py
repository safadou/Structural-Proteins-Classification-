import streamlit as st
from config import img_dir

title = "Structural Protein Classification"
sidebar_name = "Présentation"

def run():

    # TODO: choose between one of these GIFs
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    st.image(img_dir+'sequence_p.png')

    st.title(title)

    st.markdown("--------")

    with st.container():
      st.markdown("""## Qu'est ce que c'est ? """)
      st.markdown(
        """
            Les protéines sont des macromolécules organiques présentes dans toutes les cellules vivantes. 
            Elles sont les plus abondantes des molécules organiques des cellules et constituent à elles seules 
            plus de 50% du poids à sec des êtres vivants.
        """
      )
      st.markdown(""" ## A quoi ça sert?  """)
      st.markdown(
        """
            Elles remplissent de multiples fonctions pour les cellules. Elles interviennent pour des fonctions de transport (notamment d'oxygène),
            comme enzymes, ou hormones.
      """
      )
      st.image(img_dir+'proteinfunctions.png')

      st.markdown(""" ## Les classifier? Pourquoi? """)
      st.markdown(
        """
        Connaître la classe d'une protéine, revient à identifier sa fonction dans la cellule. Il est donc capital de connaître sa composition (séquence complète),
        ses propriétés physiques, dans le but de savoir à quoi elle sert dans la cellule.
        """
      )

    st.info("""
      **_Le but de notre projet est de prédire la classe d'une protéine, en nous basant sur ses propriétés physiques ou sa séquence_** 
    """ )
    #st.markdown(
    #    """
    #    Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

    #    You can browse streamlit documentation and demos to get some inspiration:
    #    - Check out [streamlit.io](https://streamlit.io)
    #    - Jump into streamlit [documentation](https://docs.streamlit.io)
    #    - Use a neural net to [analyze the Udacity Self-driving Car Image
    #      Dataset] (https://github.com/streamlit/demo-self-driving)
    #    - Explore a [New York City rideshare dataset]
    #      (https://github.com/streamlit/demo-uber-nyc-pickups)
    #    """
    #)
