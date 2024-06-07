import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Custom CSS to set the hover effect
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
[data-testid="stSidebarNavLink"] {
    background-color: #ff9a43;
    border: 3px solid #b7b7b7;
    color: #000000;
}
# [data-testid="st-emotion-cache-p5msec eqpbllx1"] {
#     background-color: ##3333FF;
#     # border: 3px solid #b7b7b7;
#     # color: #000000;
# }

[data-testid="stExpander"]{
    background-color: #3333FF;
   
    border-color : #de0034;
}

[data-testid="stMarkdown"]{
    color: #00008B;
}

[data-testid="StyledLinkIconContainer"]{
    color: #00008B;
}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(hover_css, unsafe_allow_html=True)
st.write("# PLEASE WORK")
st.write("# Clustering")
st.write("### Algorithme utilisé : K-means")
st.write("### Elbow method")


# Load data
df = pd.read_excel(r'C:\Users\ECC\Desktop\PFE\Toddler_Autism_dataset_CLEAN.xlsx')
st.write("#### nombre de k=3")

# Center and scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# K-means clustering with elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow method
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('La méthode du coude pour déterminer le nombre de clusters')
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Perform PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Perform K-means clustering with optimal clusters
kmeans_optimal = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters_optimal = kmeans_optimal.fit_predict(x_scaled)

# Add the cluster result to the original data for future analysis
df['cluster'] = clusters_optimal
st.write("## Apres K-means")

# Visualize the clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters_optimal, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title('Visualisation des clusters K-means (k=3)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
st.pyplot(fig)


# Analyze Cluster Centroids
centroids = kmeans_optimal.cluster_centers_
centroids_pca = pca.transform(centroids)

st.write("##  centres des clusters")
# Plot Centroids in PCA Space
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters_optimal, cmap='viridis', alpha=0.5)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200)
ax.set_title('Cluster Centroids in PCA Space')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
fig.colorbar(ax.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters_optimal, cmap='viridis', alpha=0.5), label='Cluster')
st.pyplot(fig)



st.write("## contenue de chaque cluster")
# Cluster Profiling
cluster_profile = df.groupby('cluster').mean()
st.write("Cluster Profiling:")
st.dataframe(cluster_profile)


with st.expander("Visualisation des profils de clusters"):
    # Visualizing Cluster Profiles
    st.write("## Visualisation des profils de clusters")
    for column in df.columns[:-1]:  # Exclude 'cluster' column
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='cluster', y=column, data=df, ax=ax)
        ax.set_title(f'{column} by Cluster')
        st.pyplot(fig)




from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



# Center and scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# Initialiser les listes pour stocker les résultats
wcss = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
intra_variances = []
inter_variances = []

# Calcul des variances intraclasse et interclasse
def intra_cluster_variance(X, labels, centroids):
    total_intra_cluster_variance = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        variance = np.sum((cluster_points - centroids[k]) ** 2)
        total_intra_cluster_variance += variance
    return total_intra_cluster_variance

def inter_cluster_variance(X, labels, centroids, global_mean):
    total_inter_cluster_variance = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        cluster_size = cluster_points.shape[0]
        variance = cluster_size * np.sum((centroids[k] - global_mean) ** 2)
        total_inter_cluster_variance += variance
    return total_inter_cluster_variance

# Calcul de la moyenne globale
global_mean = np.mean(x_scaled, axis=0)

# Tester différentes valeurs de K
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)  # Utilisation de l'initialisation par défaut
    clusters = kmeans.fit_predict(x_scaled)
    centroids = kmeans.cluster_centers_

    # Stocker les variances
    intra_variance = intra_cluster_variance(x_scaled, clusters, centroids)
    inter_variance = inter_cluster_variance(x_scaled, clusters, centroids, global_mean)

    intra_variances.append(intra_variance)
    inter_variances.append(inter_variance)

    # Stocker l'inertie (WCSS)
    wcss.append(kmeans.inertia_)

    # Calculer et stocker les indices de validation
    silhouette_avg = silhouette_score(x_scaled, clusters)
    davies_bouldin = davies_bouldin_score(x_scaled, clusters)
    calinski_harabasz = calinski_harabasz_score(x_scaled, clusters)

    silhouette_scores.append(silhouette_avg)
    davies_bouldin_scores.append(davies_bouldin)
    calinski_harabasz_scores.append(calinski_harabasz)

# Visualisation des résultats
st.write("## Méthodes de validation des clusters")

st.write("### Inertie intraclasse")
# Variance intraclasse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(2, 11), intra_variances, marker='o', color='blue')
ax.set_title('Variance intraclasse totale en fonction du nombre de clusters')
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('Variance intraclasse totale')
st.pyplot(fig)

st.write("### Inertie interclasse")
# Variance interclasse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(2, 11), inter_variances, marker='o', color='red')
ax.set_title('Variance interclasse totale en fonction du nombre de clusters')
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('Variance interclasse totale')
st.pyplot(fig)

st.write("### Indice de Silhouette")
# Indice de Silhouette
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(2, 11), silhouette_scores, marker='o', color='green')
ax.set_title('Indice de Silhouette en fonction du nombre de clusters')
ax.set_xlabel('Nombre de clusters')
ax.set_ylabel('Indice de Silhouette')
st.pyplot(fig)

# Perform PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# Perform K-means clustering with optimal clusters
kmeans_optimal = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters_optimal = kmeans_optimal.fit_predict(x_scaled)

