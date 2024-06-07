# import streamlit as st 
# import time
# st.write("# Classification")
# st.write("### Algorithme utilisé : Réseaux de neurones")
# st.write("## Performance :")

# import streamlit as st
# import pandas as pd

# # Données
# data = {
#     'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5','Fold 6'],
#     'Précision Validation 1': [0.8817, 0.9710, 0.9911, 0.9869, 1.0000],
     
#     'Précision Entraînement': [0.7427, 0.9612, 0.9887, 0.9936, 0.9963]
# }
# validation_precision = "<span style='color:blue; font-weight:bold;'>0.9661</span>"
# training_precision = "<span style='color:green; font-weight:bold;'>0.9365</span>"

# # Création du DataFrame
# df = pd.DataFrame(data)

# # Affichage du tableau dans Streamlit
# st.write("### Résultats des Précisions pour chaque Fold")
# st.dataframe(df)


# # Prediction button
# if st.button('Précision du model'):
#     # Code to perform prediction based on answers
#     # You can replace this with your prediction logic


#     # Add a placeholder
#     latest_iteration = st.empty()
#     bar = st.progress(0)

#     for i in range(98):
#     # Update the progress bar with each iteration.
        
#         bar.progress(i + 1)
#         time.sleep(0.05)
#     # Texte avec chiffres mis en évidence

#     st.markdown(f"Précision de validation sur les 5 folds est de {validation_precision}, et la précision d'entraînement sur les 5 folds est de {training_precision}.", unsafe_allow_html=True)





import streamlit as st
import pandas as pd


# Custom CSS to set the hover effect
hover_css = """
<style>
.st-emotion-cache-j7qwjs.eczjsme7:hover {
    background-color: transparent !important;
    color: black !important;
}

[data-testid="stAppViewContainer"]{
background-color: #b7cdd8;
}
[data-testid="stSidebarNavLink"] {
    background-color: #000000;
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
[data-testid="stSidebarNavLink"] {
    background-color: #ff9a43;
    border: 3px solid #b7b7b7;
    color: #000000;
}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(hover_css, unsafe_allow_html=True)


st.write("# Classification")
st.write("### Algorithme utilisé : RNN")


# Données brutes
data = {
    'Fold': [],
    'Précision Validation': [],
    'Précision Entraînement': []
}

# Ajout des données manuellement, ici pour illustration, 
# mais vous pourriez écrire un parseur si vous avez beaucoup de données
folds_data = [
    (1, 0.8693181818181818, 0.7619589977220956),
    (1, 0.9431818181818182, 0.8394077448747153),
    (1, 0.9431818181818182, 0.8766135155656796),
    (1, 0.9431818181818182, 0.8980637813211845),
    (1, 0.9545454545454546, 0.9129840546697039),
    (1, 0.9545454545454546, 0.9238800303720577),
    (1, 0.9715909090909091, 0.9324764074194598),
    (1, 0.9602272727272727, 0.9393507972665148),
    (1, 0.9772727272727273, 0.9449506454062263),
    (1, 0.9886363636363636, 0.9498861047835991),
    (1, 0.9886363636363636, 0.9540277490163594),
    (1, 0.9886363636363636, 0.9576689445709947),
    (1, 0.9943181818181818, 0.9606623444892237),
    (1, 0.9943181818181818, 0.9633908232997072),
    (1, 0.9943181818181818, 0.9657555049354594),
    (1, 0.9943181818181818, 0.9678957858769932),
    (2, 0.75, 0.662870159453303),
    (2, 0.9204545454545454, 0.7562642369020501),
    (2, 0.9431818181818182, 0.8120728929384966),
    (2, 0.9545454545454546, 0.8462414578587699),
    (2, 0.9602272727272727, 0.869248291571754),
    (2, 0.9715909090909091, 0.8861047835990888),
    (2, 0.9772727272727273, 0.9005857468272047),
    (2, 0.9772727272727273, 0.9115888382687927),
    (2, 0.9829545454545454, 0.9205264490002532),
    (2, 0.9829545454545454, 0.9279043280182232),
    (2, 0.9829545454545454, 0.9341478566991095),
    (2, 0.9886363636363636, 0.9392558845861807),
    (2, 1.0, 0.9438408971438584),
    (2, 0.9943181818181818, 0.9477709079075821),
    (2, 0.9829545454545454, 0.9511009870918755),
    (2, 0.9886363636363636, 0.9540148063781321),
    (3, 0.875, 0.7380410022779044),
    (3, 0.9488636363636364, 0.8223234624145785),
    (3, 0.9715909090909091, 0.8640850417615793),
    (3, 0.9715909090909091, 0.8883826879271071),
    (3, 0.9772727272727273, 0.9043280182232346),
    (3, 0.9829545454545454, 0.9168564920273349),
    (3, 0.9886363636363636, 0.9271070615034168),
    (3, 0.9886363636363636, 0.9350797266514806),
    (3, 0.9886363636363636, 0.9417868893950898),
    (3, 0.9829545454545454, 0.9472665148063781),
    (3, 0.9886363636363636, 0.9518533857941602),
    (3, 0.9829545454545454, 0.9555808656036446),
    (3, 0.9886363636363636, 0.9589101103907482),
    (3, 0.9886363636363636, 0.9617637487796941),
    (3, 0.9886363636363636, 0.9642369020501139),
    (3, 0.9886363636363636, 0.9664720956719818),
    (4, 0.8579545454545454, 0.7391799544419134),
    (4, 0.9090909090909091, 0.8097949886104784),
    (4, 0.9602272727272727, 0.8473804100227791),
    (4, 0.9659090909090909, 0.871867881548975),
    (4, 0.9659090909090909, 0.8881548974943052),
    (4, 0.9715909090909091, 0.9003416856492027),
    (4, 0.9772727272727273, 0.9100227790432802),
    (4, 0.9772727272727273, 0.9179954441913439),
    (4, 0.9886363636363636, 0.925208807896735),
    (4, 0.9886363636363636, 0.9316628701594533),
    (4, 0.9886363636363636, 0.9372540898736799),
    (4, 0.9943181818181818, 0.9421981776765376),
    (4, 0.9943181818181818, 0.9465568599964955),
    (4, 0.9943181818181818, 0.950374227139603),
    (4, 0.9943181818181818, 0.9536826119969628),
    (4, 0.9943181818181818, 0.9565774487471527),
    (5, 0.7828571428571428, 0.7463026166097838),
    (5, 0.9257142857142857, 0.823094425483504),
    (5, 0.92, 0.863481228668942),
    (5, 0.9428571428571428, 0.886518771331058),
    (5, 0.9657142857142857, 0.9032992036405005),
    (5, 0.9657142857142857, 0.916003033750474),
    (5, 0.9771428571428571, 0.9255647651552088),
    (5, 0.9771428571428571, 0.9330204778156996),
    (5, 0.9885714285714285, 0.9397042093287827),
    (5, 1.0, 0.9453924914675768),
    (5, 0.9942857142857143, 0.9500465404902265),
    (5, 0.9942857142857143, 0.9540197193780812),
    (5, 1.0, 0.9573816399754966),
    (5, 1.0, 0.9603445473752641),
    (5, 1.0, 0.9629882442169132),
    (5, 1.0, 0.9652303754266212),
    (6, 0.84, 0.8350398179749715),
    (6, 0.92, 0.8754266211604096),
    (6, 0.9485714285714286, 0.8968524838832006),
    (6, 0.9657142857142857, 0.9135381114903299),
    (6, 0.9771428571428571, 0.9258248009101251),
    (6, 0.9771428571428571, 0.9351535836177475),
    (6, 0.9828571428571429, 0.9429546562652364),
    (6, 0.9828571428571429, 0.9493742889647326),
    (6, 0.9828571428571429, 0.9546201491593983),
    (6, 0.9828571428571429, 0.9588168373151308),
    (6, 0.9828571428571429, 0.962457337883959),
    (6, 0.9885714285714285, 0.9654910883579826),
    (6, 0.9828571428571429, 0.9680581079898486),
    (6, 0.9828571428571429, 0.9702584105314481),
    (6, 0.9885714285714285, 0.9721653394008343),
    (6, 0.9828571428571429, 0.9739050056882821),
]

for fold, val_prec, train_prec in folds_data:
    data['Fold'].append(fold)
    data['Précision Validation'].append(val_prec)
    data['Précision Entraînement'].append(train_prec)

# Création du DataFrame
df = pd.DataFrame(data)

# Affichage dans Streamlit
st.write("Précision Validation et Entraînement par Fold")
st.dataframe(df)

# Calcul des moyennes
moyennes = df.groupby('Fold').mean().reset_index()
moyennes['Moyenne Précision Validation'] = moyennes['Précision Validation']
moyennes['Moyenne Précision Entraînement'] = moyennes['Précision Entraînement']
moyennes = moyennes[['Fold', 'Moyenne Précision Validation', 'Moyenne Précision Entraînement']]

st.write("Moyennes par Fold")
st.dataframe(moyennes)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

# Charger les données en ignorant la colonne Unnamed: 0 et réindexer les lignes
df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset_CLEAN.xlsx').reset_index(drop=True)

# Fix the random seed for reproducibility
torch.manual_seed(42)

# Séparer les caractéristiques et les étiquettes
X = df.drop(columns=['ASD'])
Y = df['ASD']

class MyRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Première couche cachée (RNN)
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        nn.init.xavier_uniform_(self.rnn1.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn1.weight_hh_l0)
        nn.init.zeros_(self.rnn1.bias_ih_l0)
        nn.init.zeros_(self.rnn1.bias_hh_l0)

        # Deuxième couche cachée (RNN)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        nn.init.xavier_uniform_(self.rnn2.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn2.weight_hh_l0)
        nn.init.zeros_(self.rnn2.bias_ih_l0)
        nn.init.zeros_(self.rnn2.bias_hh_l0)

        # Couche entièrement connectée
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # Fonction d'activation sigmoïde
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        # Initialisation des états cachés pour les deux couches RNN
        h0_1 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        h0_2 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # Passage par la première couche RNN
        out1, _ = self.rnn1(x.view(batch_size, -1, self.input_size), h0_1)
        out1 = F.relu(out1)

        # Passage par la deuxième couche RNN
        out2, _ = self.rnn2(out1, h0_2)
        out2 = F.relu(out2)

        # Passage par la couche entièrement connectée et application de l'activation sigmoïde
        out = self.fc(out2[:, -1, :])
        out = self.sigmoid(out)

        return out

# Définir les hyperparamètres
num_epochs = 16
learning_rate = 0.001
n_splits = 6

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()

# Définir la validation croisée K-Folds
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialisation des listes pour stocker les précisions de validation et d'entraînement de chaque fold
all_val_accuracies = []
all_train_accuracies = []

# Liste de couleurs pour chaque fold
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Boucle de validation croisée K-Folds
for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    print(f"Fold {fold+1}/{n_splits}")

    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    Y_train_fold, Y_val_fold = Y.iloc[train_index], Y.iloc[val_index]

    # Convertir les données en tenseurs PyTorch
    X_train_fold_tensor = torch.tensor(X_train_fold.values, dtype=torch.float32)
    Y_train_fold_tensor = torch.tensor(Y_train_fold.values, dtype=torch.float32).view(-1, 1)
    X_val_fold_tensor = torch.tensor(X_val_fold.values, dtype=torch.float32)
    Y_val_fold_tensor = torch.tensor(Y_val_fold.values, dtype=torch.float32).view(-1, 1)

    # Préparer les DataLoader pour les folds
    train_dataset = TensorDataset(X_train_fold_tensor, Y_train_fold_tensor)
    val_dataset = TensorDataset(X_val_fold_tensor, Y_val_fold_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialiser le modèle
    model = MyRNNModel(input_size=X_train_fold_tensor.shape[1], hidden_size=56, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_correct = 0
    train_total = 0
    train_total_loss = 0

    # Boucle d'entraînement sur plusieurs époques
    for epoch in range(num_epochs):
        model.train() # Passage du modèle en mode entraînement
        train_total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad() # Remise à zéro des gradients
            outputs = model(inputs) # Passage des données d'entraînement dans le modèle
            loss = criterion(outputs.squeeze(), labels.squeeze()) # Calcul de la perte d'entraînement
            loss.backward() # Rétropropagation pour calculer les gradients
            optimizer.step() # Mise à jour des poids
            train_total_loss += loss.item() # Ajout de la perte du batch à la perte totale
            # Calcul du nombre de prédictions correctes et du nombre total d'éléments pour l'entraînement sur le batch actuel
            train_correct += (outputs.squeeze() > 0.5).eq(labels.squeeze()).sum().item()
            train_total += labels.size(0) # train_total correspond au nombre total d'éléments dans le batch d'entraînement

        train_accuracy = train_correct / train_total  # Calcul de la précision d'entraînement pour cette epoch
        all_train_accuracies.append(train_accuracy) # Stocker la précision d'entraînement pour cette epoch

        # Évaluation sur les données de validation
        model.eval()# Passage du modèle en mode évaluation
        val_correct = 0
        val_total = 0
        val_total_loss = 0
        with torch.no_grad(): # Désactivation du calcul des gradients pour l'évaluation
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Passage des données de validation dans le modèle
                predicted = (outputs.squeeze() > 0.5).float()  # Application du seuil de décision (0.5)
                # Calcul du nombre de prédictions correctes et du nombre total d'éléments pour la validation sur le batch actuel
                val_total += labels.size(0)
                val_correct += (predicted == labels.squeeze()).sum().item()
                val_total_loss += criterion(outputs.squeeze(), labels.squeeze()).item() # val_total correspond au nombre total d'éléments dans le batch de validation

        # Calculer et afficher la précision de validation pour ce fold
        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}, Précision Entraînement: {train_accuracy:.4f}, Précision Validation: {val_accuracy:.4f}")
        all_val_accuracies.append(val_accuracy) # Stocker la précision de validation pour cette epoch

    # Calcul de la moyenne de ce fold
    fold_val_accuracy = sum(all_val_accuracies[-num_epochs:]) / num_epochs
    fold_train_accuracy = sum(all_train_accuracies[-num_epochs:]) / num_epochs
    print(f"Moyenne Précision Validation Fold {fold+1}: {fold_val_accuracy}")
    print(f"Moyenne Précision Entraînement Fold {fold+1}: {fold_train_accuracy}")
# Calcul des moyennes générales des précisions de validation et d'entraînement sur tous les folds
average_val_accuracy = sum(all_val_accuracies) / (num_epochs * n_splits)
average_train_accuracy = sum(all_train_accuracies) / (num_epochs * n_splits)

st.write(f"Moyenne générale Précision Entraînement: {average_train_accuracy:.4f}")

st.write(f"Moyenne générale Précision Validation: {average_val_accuracy:.4f}")

# Liste de couleurs pour chaque fold
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# # Tracé des graphiques de précision pour l'entraînement
# plt.figure(figsize=(15, 5))
# for fold in range(n_splits):
#     plt.plot(range(1, num_epochs+1), all_train_accuracies[fold*num_epochs:(fold+1

# Tracé des graphiques de précision pour l'entraînement
plt.figure(figsize=(15, 5))
for fold in range(n_splits):
    plt.plot(range(1, num_epochs+1), all_train_accuracies[fold*num_epochs:(fold+1)*num_epochs], label=f'Fold {fold+1}', color=colors[fold])
plt.axhline(y=1.0, color='k', linestyle='--', label='Maximum (1.0)')
plt.xlabel('Époques')
plt.ylabel('Précision d\'Entraînement')
plt.title('Précision d\'Entraînement par époque pour chaque fold')
plt.legend()

# Utilisez Streamlit pour afficher le graphique
st.pyplot(plt)

# Tracé des graphiques de précision pour la validation
plt.figure(figsize=(15, 5))
for fold in range(n_splits):
    plt.plot(range(1, num_epochs+1), all_val_accuracies[fold*num_epochs:(fold+1)*num_epochs], label=f'Fold {fold+1}', linestyle='--', color=colors[fold])
plt.axhline(y=1.0, color='k', linestyle='--', label='Maximum (1.0)')
plt.xlabel('Époques')
plt.ylabel('Précision de Validation')
plt.title('Précision de Validation par époque pour chaque fold')
plt.legend()

# Utilisez Streamlit pour afficher le graphique
st.pyplot(plt)

# Calcul des moyennes des précisions d'entraînement et de validation pour chaque fold
fold_avg_train_accuracies = [sum(all_train_accuracies[i*num_epochs:(i+1)*num_epochs]) / num_epochs for i in range(n_splits)]
fold_avg_val_accuracies = [sum(all_val_accuracies[i*num_epochs:(i+1)*num_epochs]) / num_epochs for i in range(n_splits)]

# Tracé de l'histogramme des précisions moyennes d'entraînement pour chaque fold
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(n_splits)

# Tracer les barres pour les précisions moyennes d'entraînement
bar1 = plt.bar(index, fold_avg_train_accuracies, bar_width, label='Entraînement', color='blue')

# Ajout des annotations des valeurs des barres pour les précisions moyennes d'entraînement
for i in range(len(index)):
    plt.text(i, fold_avg_train_accuracies[i] + 0.01, f'{fold_avg_train_accuracies[i]:.2f}', ha='center', color='blue')

# Ajout d'une ligne rouge pour le maximum des précisions moyennes
max_accuracy = 1
plt.axhline(y=max_accuracy, color='red', linestyle='--', linewidth=2, label='Max Précision (1.0)')

plt.xlabel('Fold')
plt.ylabel('Précision Moyenne')
plt.title('Précisions Moyennes d\'Entraînement pour chaque Fold')
plt.xticks(index, [f'Fold {i+1}' for i in index])
plt.legend()

# Utilisez Streamlit pour afficher le graphique
st.pyplot(plt)

# Tracé de l'histogramme des précisions moyennes de validation pour chaque fold
plt.figure(figsize=(10, 6))

# Tracer les barres pour les précisions moyennes de validation
bar2 = plt.bar(index, fold_avg_val_accuracies, bar_width, label='Validation', color='green')

# Ajout des annotations des valeurs des barres pour les précisions moyennes de validation
for i in range(len(index)):
    plt.text(i, fold_avg_val_accuracies[i] + 0.01, f'{fold_avg_val_accuracies[i]:.2f}', ha='center', color='green')

# Ajout d'une ligne rouge pour le maximum des précisions moyennes
max_accuracy = 1
plt.axhline(y=max_accuracy, color='red', linestyle='--', linewidth=2, label='Max Précision (1.0)')

plt.xlabel('Fold')
plt.ylabel('Précision Moyenne')
plt.title('Précisions Moyennes de Validation pour chaque Fold')
plt.xticks(index, [f'Fold {i+1}' for i in index])
plt.legend()

# Utilisez Streamlit pour afficher le graphique
st.pyplot(plt)
