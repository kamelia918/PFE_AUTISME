# import streamlit as st 

# # Custom CSS to set the background color
# page_bg_img = """
# <style>
# [data-testid="stAppViewContainer"]{
# background-color: #00002c;
# opacity: 0.8;
# background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #00002c 10px ), repeating-linear-gradient( #dddfff55, #dddfff );}
# [data-testid="stHeader"]{
# background: rgb(0 0 0 / 0%);
# }
# [data-testid="stSidebarNavLink"]:hover {
#     background-color: #e5e5f7;
#     text-color: #000000;
# }

# [data-testid="bonne-exploration"]{
#     background-color: #e5e5f7;
#     color: #000000;
# }

# # [data-testid="stMarkdownContainer"]{
# #     color: #00002c;
# #     font-weight:500;
# # }

# [data-testid="stExpander"]{
#     background-color: #00002c;
#     border-radius:5%;
# }

# </style>
# """

# # Inject CSS into the Streamlit app
# st.markdown(page_bg_img, unsafe_allow_html=True)

# st.title("Bienvenue sur l'Application de Prédiction d'Autisme")

# with st.expander("### À propos de cette application"):
#     st.write("""

#     Cette application est conçue pour aider à l'analyse et à la prédiction du trouble du spectre de l'autisme (TSA) en utilisant des algorithmes de Machine Learning et d'Intelligence Artificielle. Notre outil offre plusieurs fonctionnalités puissantes pour traiter et analyser les données afin de fournir des résultats précis et utiles.


#     """)


# with st.expander("Fonctionnalités de l'application"):
#     st.write("""


#     - **Prétraitement des données (Preprocessing)** : Cette section explique et montre comment les données sont nettoyées et préparées avant d'être utilisées dans les modèles.
            
#     - **K-Means Clustering** : Découvrez comment le clustering K-means est utilisé pour identifier des groupes distincts dans les données. Vous pourrez visualiser les clusters formés et comprendre leur signification.
            
#     - **Réseaux de Neurones Récurentiels (RNN)** : Explorez l'utilisation des réseaux de neurones récurrents pour la prédiction de l'autisme. Cette section présente les détails techniques, les performances et les résultats obtenus avec les RNN.

#     - **Tester un Cas Réel** : Utilisez cette fonctionnalité pour tester l'application avec des données réelles. Entrez les informations nécessaires et laissez l'application prédire si le cas est atteint de TSA ou non.


#     """)


# with st.expander("Comment utiliser cette application"):
#     st.write("""


#     1. **Explorez les sections** : Parcourez chaque section pour comprendre les différentes étapes de l'analyse et de la prédiction.
#     2. **Testez un cas réel** : Utilisez la section dédiée pour entrer des données spécifiques et obtenir une prédiction.
#     3. **Consultez les résultats** : Obtenez des visualisations et des rapports détaillés pour chaque méthode utilisée.

    
#     """)

# st.write("Nous espérons que cette application vous sera utile pour mieux comprendre et prédire le trouble du spectre de l'autisme. Si vous avez des questions ou des suggestions, n'hésitez pas à nous contacter.")
# st.write(" ### Bonne exploration !")



import base64
from pathlib import Path
import streamlit as st 

# Custom CSS to set the background color
def get_base64_image(image_path):
    img_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return f"data:image/JPG;base64,{encoded}"

# Chemin de votre image locale
image_path = r"C:\Users\ECC\Desktop\PFE\autisme3.jpg"
bg_img = get_base64_image(image_path)



# CSS personnalisé avec l'image de fond
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]{{
opacity: 1;
background-image: url('{bg_img}');
background-size: cover;
background-blend-mode: overlay;
}}

[data-testid="stHeader"]{{
background: rgb(0 0 0 / 0%);
}}

[data-testid="stSidebarNavLink"]:hover {{
    background-color: #000000;
    color: #e0643a;
         font-weight: 300;

}}

[data-testid="bonne-exploration"]{{
    background-color: #000000;
    color: #0d0103;
     font-weight: 300;

}}

[data-testid="stMarkdownContainer"]{{
    color: #0d0103;
    font-weight: 500;
    border-radius: 3%;

}}

[data-testid="stExpander"]{{
    background-color: #ff863b;
    border-radius: 3%;
    border-color : #de0034;
}}

[data-testid="stSidebarContent"]{{

opacity: 2;
background-image: url('{bg_img}');
background-size: cover;
background-blend-mode: overlay;
}}

[data-testid="stSidebarContent"]{{

background-color:#b7b7b7;
border:3px solid #ff9a43;
}}

[data-testid="baseButton-header"]{{

background-color:#b7b7b7;
border:3px solid #ff9a43;
}}

[data-testid="baseButton-headerNoPadding"]{{

background-color:#b7b7b7;
border:3px solid #ff9a43;
}}

[data-testid="StyledLinkIconContainer"]{{
color:#000000;
}}

[data-testid="stSidebarNavLink"] {{
    background-color: #ff9a43 ;
    border: 3px solid #b7b7b7;
    color: #000000;
}}


</style>
"""
# Inject CSS into the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Bienvenue sur l'Application de Prédiction d'Autisme")

with st.expander("### À propos de cette application"):
    st.write("""

    Cette application est conçue pour aider à l'analyse et à la prédiction du trouble du spectre de l'autisme (TSA) en utilisant des algorithmes de Machine Learning et d'Intelligence Artificielle. Notre outil offre plusieurs fonctionnalités puissantes pour traiter et analyser les données afin de fournir des résultats précis et utiles.


    """)


with st.expander("Fonctionnalités de l'application"):
    st.write("""


    - **Prétraitement des données (Preprocessing)** : Cette section explique et montre comment les données sont nettoyées et préparées avant d'être utilisées dans les modèles.
            
    - **K-Means Clustering** : Découvrez comment le clustering K-means est utilisé pour identifier des groupes distincts dans les données. Vous pourrez visualiser les clusters formés et comprendre leur signification.
            
    - **Réseaux de Neurones Récurentiels (RNN)** : Explorez l'utilisation des réseaux de neurones récurrents pour la prédiction de l'autisme. Cette section présente les détails techniques, les performances et les résultats obtenus avec les RNN.

    - **Tester un Cas Réel** : Utilisez cette fonctionnalité pour tester l'application avec des données réelles. Entrez les informations nécessaires et laissez l'application prédire si le cas est atteint de TSA ou non.


    """)


with st.expander("Comment utiliser cette application"):
    st.write("""


    1. **Explorez les sections** : Parcourez chaque section pour comprendre les différentes étapes de l'analyse et de la prédiction.
    2. **Testez un cas réel** : Utilisez la section dédiée pour entrer des données spécifiques et obtenir une prédiction.
    3. **Consultez les résultats** : Obtenez des visualisations et des rapports détaillés pour chaque méthode utilisée.

    
    """)

st.write("Nous espérons que cette application vous sera utile pour mieux comprendre et prédire le trouble du spectre de l'autisme. Si vous avez des questions ou des suggestions, n'hésitez pas à nous contacter.")
st.write(" ### Bonne exploration !")
