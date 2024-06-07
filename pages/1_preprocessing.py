import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.write("# Preprocessing")


# Custom CSS to set the hover effect and background image
hover_css = """
<style>
[data-testid="stAppViewContainer"]{
background-color: #b7cdd8;
}
.st-emotion-cache-j7qwjs.eczjsme7:hover {
    background-color: transparent !important;
    color: black !important;
}

[data-testid="stSidebarContent"]{

opacity: 2;
background-image: url('{bg_img}');
background-size: cover;
background-blend-mode: overlay;
}

[data-testid="stSidebarContent"]{

background-color:#b7b7b7;
border:3px solid #ff9a43;
}
[data-testid="stSidebarNavLink"] {
    background-color: #000000;
}
[data-testid="stHeader"] {
    background-color: #b7cdd8;
}
[data-testid="stMarkdownContainer"]{
    color: #0d0103;
    font-weight: 500;
    border-radius: 3%;
}
[data-testid="stExpander"]{
    background-color: #ff863b;
    /* border-color : #de0034; */ /* This comment seems misplaced; if needed, adjust the border-color */
}
[data-testid="baseButton-header"] {
    background-color: #b7b7b7;
    border: 3px solid #ff9a43;
}
[data-testid="baseButton-headerNoPadding"] {
    background-color: #b7b7b7;
    border: 3px solid #ff9a43;
}
[data-testid="StyledLinkIconContainer"] {
    color: #000000;
}
[data-testid="stSidebarNavLink"] {
    background-color: #ff9a43;
    border: 3px solid #b7b7b7;
    color: #000000;
}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(hover_css, unsafe_allow_html=True)

with st.expander("Base de donnée"):
    st.write("We provide in this section more information about our database.")
    df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset.xlsx')

    columns_info = {
        'Column Name': df.columns,
        'Explanation': [
            "numero de ligne",
            "Est-ce que votre enfant vous regarde quand vous appelez son nom ?",
            "Est-ce facile pour vous d'établir un contact visuel avec votre enfant ?",
            "Est-ce que votre enfant pointe du doigt pour indiquer qu’il veut quelque chose ? (par exemple, un jouet hors de portée)",
            "Est-ce que votre enfant pointe du doigt pour partager son intérêt avec vous ? (par exemple, en montrant un objet intéressant)",
            "Est-ce que votre enfant fait semblant ? (par exemple, s’occuper de poupées, parler au téléphone-jouet)",
            "Est-ce que votre enfant suit du regard ou vous regarde ?",
            "Si vous ou quelqu’un d’autre dans la famille êtes visiblement contrarié, est-ce que votre enfant montre des signes de vouloir vous réconforter ? (par exemple, en caressant les cheveux, en les serrant dans ses bras)",
            "Décrivez-vous les premiers mots de votre enfant comme inhabituels ?",
            "Est-ce que votre enfant utilise des gestes simples ? (par exemple, dire au revoir de la main)",
            "Est-ce que votre enfant fixe un point sans but apparent ?",
            "Représente l’âge de l’enfant en mois (de 12 à 36).",
            "Les scores totaux vont de 0 à 10.",
            "Fille ou garçon.",
            "Les origines.",
            "Si il est atteint ou pas de jaunisse.",
            "Membre de la famille souffrant de l’autisme.",
            "Qui a complété les 10 questions.",
            "Détermine si l'enfant souffre d'autisme."
        ]
    }

    columns_df = pd.DataFrame(columns_info)
    st.write("## Tableau des colonnes et leurs explications")
    st.write(columns_df)

    if 'Ethnicity' in df.columns:
        st.write("## Description de chaque variable")
        st.write("### Ethnicity")

        labels = [
            'White European', 'Asian', 'Middle Eastern', 'South Asian', 'Black',
            'Hispanic', 'Others', 'Latino', 'Pacifica', 'Mixed', 'Native Indian'
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(df['Ethnicity'].value_counts(), autopct='%1.1f%%', textprops={'fontsize': 12})
        ax.legend(title="Ethnicity:", labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        st.pyplot(fig)
        st.write("D’après nos données, le pourcentage de “Blanc Européen” est le plus élevé.")

        st.image(r'C:\Users\ECC\Desktop\PFE\images\SEXE.png', caption='Elbow Method', use_column_width=True)
        st.image(r'C:\Users\ECC\Desktop\PFE\images\AGE.png', caption='Elbow Method', use_column_width=True)
        st.image(r'C:\Users\ECC\Desktop\PFE\images\QA_SCORE.png', caption='Elbow Method', use_column_width=True)
        st.image(r'C:\Users\ECC\Desktop\PFE\images\FAMILLE_ATTIENT.png', caption='Elbow Method', use_column_width=True)
        st.image(r'C:\Users\ECC\Desktop\PFE\images\WHO_COMPLETED_THE_TEST.png', caption='Elbow Method', use_column_width=True)
    else:
        st.write("La colonne 'Ethnicity' n'existe pas dans le DataFrame.")

    st.write("### Conclusion :")
    st.write("D’après nos données, le nombre maximum d’enfants souffrant d’autisme ne s’inquiète pas lorsque leurs proches sont contrariés. Cela montre que la plupart des patients autistes présentent un manque d’émotion.")

    st.write("### Conclusion de l’analyse :")
    st.write("1. Les garçons sont plus susceptibles d’avoir un TSA que les filles.")
    st.write("2. Les enfants de 36 mois ont le plus grand nombre de cas de TSA dans le monde.")
    st.write("3. Les enfants de 2 ans sont plus susceptibles d’avoir un TSA.")
    st.write("4. Le TSA n’est pas une maladie héréditaire.")
    st.write("5. Les enfants ayant eu un ictère sont plus susceptibles d’avoir un TSA que les tout-petits normaux.")

with st.expander("Preprocessing"):
    st.write('## Travail fait : ')
    st.write('EX : equilibrer la base de données ..... ')
    st.write('## avant preprocessing')
    df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset.xlsx')
    st.write(df)

    st.write('## Apres preprocessing')
    df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset_CLEAN.xlsx')
    st.write(df)
