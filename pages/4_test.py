# import streamlit as st 
# import time
# import joblib

# # Custom CSS to set the hover effect
# hover_css = """
# <style>
# .st-emotion-cache-j7qwjs.eczjsme7:hover {
#     background-color: transparent !important;
#     color: black !important;
# }

# [data-testid="stSidebarNavLink"] {
#     background-color: #000000;
# }

# [data-testid="stAppViewContainer"]{
# background-color: #b7cdd8;
# }
# [data-testid="stSidebarNavLink"] {
#     background-color: #000000;
# }

# [data-testid="stSidebarContent"]{

# opacity: 2;
# background-image: url('{bg_img}');
# background-size: cover;
# background-blend-mode: overlay;
# }

# [data-testid="stSidebarContent"]{

# background-color:#b7b7b7;
# border:3px solid #ff9a43;
# }
# [data-testid="stSidebarNavLink"] {
#     background-color: #000000;
# }
# [data-testid="stHeader"] {
#     background-color: #b7cdd8;
# }
# [data-testid="stSidebarNavLink"] {
#     background-color: #ff9a43;
#     border: 3px solid #b7b7b7;
#     color: #000000;
# }
# [data-testid="stMarkdownContainer"] {
    
#     color: #000000;
# }
# [data-testid="baseButton-secondary"] {
    
#     background-color: #ff9a43;
# }

# </style>
# """

# # Inject CSS into the Streamlit app
# st.markdown(hover_css, unsafe_allow_html=True)



# # Sauvegarder le modèle KNN
# # joblib.dump(knn_model, 'knn_model.joblib')


# st.write("# TEST")
# st.write("## General informations")

# st.selectbox('Sélectionnez le genre', ['Male', 'Femelle'])
# st.selectbox(' diagnostiqué avec', ['oui', 'non'])
# st.selectbox('A-t-il un membre de la famille atteint d autisme', ['oui', 'non'])
# st.selectbox('Age', ['1', '2','3','4','5'])

# st.write("## Questions")

# questions = [
#     "Est-ce que votre enfant vous regarde quand vous l'appelez par son nom?",
#     "Est-il facile pour vous d'établir un contact visuel avec votre enfant?",
#     "Est-ce que votre enfant pointe du doigt pour indiquer ce qu'il veut?",
#     "Est-ce que votre enfant pointe du doigt pour partager son intérêt avec vous?",
#     "Est-ce que votre enfant joue à faire semblant?",
#     "Est-ce que votre enfant suit votre regard?",
#     "Si vous ou quelqu'un d'autre dans la famille êtes visiblement contrarié, est-ce que votre enfant montre des signes de vouloir réconforter?",
#     "Décririez-vous les premiers mots de votre enfant comme inhabituels?",
#     "Est-ce que votre enfant utilise des gestes simples?",
#     "Est-ce que votre enfant fixe des points sans but apparent?",
#     "Est-ce que vous ou quelqu'un d'autre dans la famille êtes atteint d'autisme?",
#     "Est-ce que votre enfant a souffert de jaunisse après la naissance?"
# ]

# # Add questions and checkboxes to the frame
# for i, question in enumerate(questions):
#     st.selectbox(question, ['oui', 'non'])


# # Prediction button
# if st.button('Prédire'):
#     # Code to perform prediction based on answers
#     # You can replace this with your prediction logic


#     # Add a placeholder
#     latest_iteration = st.empty()
#     bar = st.progress(0)

#     for i in range(100):
#     # Update the progress bar with each iteration.
        
#         bar.progress(i + 1)
#         time.sleep(0.01)
  
#     st.write("Prédiction effectuée !")
#     st.write("Résuletat : autiste")























import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random

# Load the dataset
df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset_CLEAN.xlsx').reset_index(drop=True)

# Separate features and labels
X = df.drop(columns=['ASD'])
Y = df['ASD']

# Define the model
class MyRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        h0_1 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        h0_2 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out1, _ = self.rnn1(x.view(batch_size, -1, self.input_size), h0_1)
        out1 = F.relu(out1)
        out2, _ = self.rnn2(out1, h0_2)
        out2 = F.relu(out2)
        out = self.fc(out2[:, -1, :])
        out = self.sigmoid(out)
        return out

# Load the trained model
input_size = 10  # Update this value based on your training configuration
model = MyRNNModel(input_size=input_size, hidden_size=56, output_size=1)
model.load_state_dict(torch.load(r'C:\Users\ECC\Desktop\PFE\model.pth'))  # Ensure your model is saved after training

# Streamlit app
st.title('ASD Prediction App')

st.write("This app predicts the likelihood of Autism Spectrum Disorder (ASD) based on input features.")

# Randomly select an example from the dataset
index = random.randint(0, len(X) - 1)
input_data = X.iloc[index]
expected_output = Y.iloc[index]

# Convert to PyTorch tensor
input_tensor = torch.tensor(input_data.values, dtype=torch.float32).view(1, -1)

# Prediction
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    prediction = (output > 0.5).item()

# Display input features
st.subheader('Input Features')
input_with_titles = "\n".join([f"{column}: {value}" for column, value in zip(X.columns, input_data)])
st.text(input_with_titles)

# Display the actual and predicted output
st.subheader('Output')
st.write(f"Expected Output (ASD): {'Yes' if expected_output == 1 else 'No'}")
st.write(f"Predicted Output (ASD): {'Yes' if prediction else 'No'}")


