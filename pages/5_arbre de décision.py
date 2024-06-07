import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import export_graphviz
import graphviz

# Charger les données
df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset_CLEAN.xlsx')
df.replace(('m', 'f'), (1, 0), inplace=True)
X = df.drop(columns=['ASD', 'QA_Score', 'Ethnicity'])
y = df['ASD']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Créer et entraîner le modèle d'arbre de décision
model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
model.fit(X_train, Y_train)

# Faire des prédictions sur l'ensemble de test et d'entraînement
Y_hat_test = model.predict(X_test).astype(int)
Y_hat_train = model.predict(X_train).astype(int)

# Analyser les résultats
def analysis_of_results(model, model_name, X_test, Y_test, Y_hat_test, Y_hat_train):
    # Calculer les performances sur l'ensemble de test
    accuracy = accuracy_score(Y_test, Y_hat_test)
    precision = precision_score(Y_test, Y_hat_test)
    recall = recall_score(Y_test, Y_hat_test)
    f1 = f1_score(Y_test, Y_hat_test)
    
    # Afficher les performances
    st.write(f"# Performance du modèle {model_name} :")
    st.write(f"Précision : {precision}")
    st.write(f"Rappel : {recall}")
    st.write(f"Score F1 : {f1}")

analysis_of_results(model, 'Decision Tree', X_test, Y_test, Y_hat_test, Y_hat_train)

st.write("## Histogrammes des features les plus importantes")
# Fonction pour afficher les histogrammes des features les plus importantes
def plot_feature_importance(model, X_train, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.title("Feature importances")
    for i in range(X_train.shape[1]):
        color = 'blue' if X_train.columns[indices[i]] == 'autiste' else 'red'
        plt.bar(i, importances[indices[i]], color=color)
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation='vertical')
    plt.xlim([-1, X_train.shape[1]])
    st.pyplot(plt.gcf())

plot_feature_importance(model, X_train, X_train.columns)

st.write("## Arbre de décision")
# Exporter l'arbre de décision au format DOT
dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns,
                           class_names=['Non-Autiste', 'Autiste'], filled=True, rounded=True,
                           special_characters=True)

# Afficher l'arbre avec graphviz
graph = graphviz.Source(dot_data)
st.graphviz_chart(dot_data)
