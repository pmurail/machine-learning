import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


# Générer et sauvegarder le fichier CSV
# Lire le fichier CSV
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Prétraiter les données
def preprocess_data(data):
    # Nettoyage des commentaires
    stop_words = stopwords.words('french')
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_df=0.85)
    X = vectorizer.fit_transform(data['Comment'])
    return X, vectorizer.get_feature_names_out()


# Appliquer K-means clustering
def perform_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans


# Afficher les résultats
def display_results(data, kmeans, X):
    data['Cluster'] = kmeans.labels_

    # Réduire la dimension pour la visualisation avec t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(X.toarray())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=data['Cluster'], cmap='viridis')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('K-means Clustering des commentaires')
    plt.colorbar(scatter, ticks=[0, 1, 2])
    plt.clim(-0.5, 2.5)

    plt.show()

    # Afficher les commentaires par cluster
    cluster_labels = ["positif", "neutre", "negatif"]  # À ajuster manuellement selon les résultats
    for cluster in range(3):
        print(f"\nGroupe {cluster + 1} ({cluster_labels[cluster]}) :")
        cluster_comments = data[data['Cluster'] == cluster]['Comment'].tolist()
        for comment in cluster_comments:
            print(f"- {comment}")

    # Calculer le score de silhouette
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f"Score de silhouette : {silhouette_avg}")


# Fonction principale
def main(file_path, n_clusters=3):
    data = load_data(file_path)
    processed_data, feature_names = preprocess_data(data)
    kmeans = perform_clustering(processed_data, n_clusters)
    display_results(data, kmeans, processed_data)


# Exécution du script
if __name__ == "__main__":
    file_path = "comments.csv"  # Chemin de votre fichier CSV
    main(file_path, n_clusters=3)  # Vous pouvez changer le nombre de clusters si nécessaire
