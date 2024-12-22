import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils import shuffle

# Load the dataset
data = pd.read_csv('dataset.csv')  

genre_mapping = {
    # Pop
    'pop': 'Pop', 'power-pop': 'Pop', 'pop-film': 'Pop', 'indie-pop': 'Pop',
    # Rock
    'rock': 'Rock', 'alt-rock': 'Rock', 'punk-rock': 'Rock', 'hard-rock': 'Rock',
    'psych-rock': 'Rock', 'grunge': 'Rock', 'metal': 'Rock', 'heavy-metal': 'Rock',
    'punk': 'Rock', 'rockabilly': 'Rock',
    # Electronic
    'edm': 'Electronic', 'house': 'Electronic', 'electro': 'Electronic', 'dubstep': 'Electronic',
    'techno': 'Electronic', 'minimal-techno': 'Electronic', 'progressive-house': 'Electronic',
    'detroit-techno': 'Electronic', 'deep-house': 'Electronic', 'chicago-house': 'Electronic',
    # Jazz/Blues
    'jazz': 'Jazz/Blues', 'blues': 'Jazz/Blues', 'bluegrass': 'Jazz/Blues',
    'funk': 'Jazz/Blues', 'soul': 'Jazz/Blues', 'gospel': 'Jazz/Blues',
    # Classical/Instrumental
    'classical': 'Classical/Instrumental', 'opera': 'Classical/Instrumental',
    'piano': 'Classical/Instrumental', 'instrumental': 'Classical/Instrumental', 'ambient': 'Classical/Instrumental',
    # Hip-Hop/Rap
    'hip-hop': 'Hip-Hop/Rap', 'rap': 'Hip-Hop/Rap', 'trap': 'Hip-Hop/Rap',
    # World/Folk
    'latin': 'World/Folk', 'brazil': 'World/Folk', 'afrobeat': 'World/Folk',
    'salsa': 'World/Folk', 'reggae': 'World/Folk', 'folk': 'World/Folk',
    'country': 'World/Folk', 'mandopop': 'World/Folk', 'cantopop': 'World/Folk',
    'k-pop': 'World/Folk', 'j-pop': 'World/Folk', 'turkish': 'World/Folk', 'indian': 'World/Folk',
    # Dance/Party
    'dance': 'Dance/Party', 'club': 'Dance/Party', 'party': 'Dance/Party',
    'dancehall': 'Dance/Party', 'drum-and-bass': 'Dance/Party',
    # Miscellaneous
    'anime': 'Miscellaneous', 'children': 'Miscellaneous',
    'comedy': 'Miscellaneous', 'study': 'Miscellaneous',
    'happy': 'Miscellaneous', 'sad': 'Miscellaneous',
    'disco': 'Miscellaneous', 'disney': 'Miscellaneous',
    'show-tunes': 'Miscellaneous', 'songwriter': 'Miscellaneous',
    'j-idol': 'Miscellaneous', 'tango': 'Miscellaneous',
    'grindcore': 'Miscellaneous', 'new-age': 'Miscellaneous',
    'other': 'Miscellaneous', 'other/miscellaneous': 'Miscellaneous'
}

data['broad_genre'] = data['track_genre'].map(genre_mapping).fillna('Miscellaneous')

# Inspect the dataset
print(data.head())

if 'broad_genre' in data.columns and 'popularity' in data.columns:
    le = LabelEncoder()
    data['genre_encoded'] = le.fit_transform(data['broad_genre'])

    features = data[['genre_encoded', 'popularity','acousticness','liveness']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=5, random_state=51)  
    data['cluster'] = kmeans.fit_predict(scaled_features)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(cluster_data['genre_encoded'], cluster_data['popularity'], label=f'Cluster {cluster}')
    
    cluster_data.to_csv('clustered_data.csv', index=False)

    # Label the axes and title
    plt.xticks(ticks=range(len(le.classes_)), labels=le.classes_, rotation=90)
    plt.xlabel('Genre')
    plt.ylabel('Popularity')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.tight_layout()
    plt.show()

    genre_popularity = data.groupby('broad_genre')['popularity'].mean().sort_values(ascending=False)
    print("Average Popularity by Genre:")
    print(genre_popularity)

    plt.figure(figsize=(12, 6))
    genre_popularity.plot(kind='bar')
    plt.xlabel('Genre')
    plt.ylabel('Average Popularity')
    plt.title('Average Popularity of Songs by Genre')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    data['true_labels'] = le.transform(data['broad_genre'])
    
    
    cluster_labels = data['cluster']
    true_labels = data['true_labels']

    cluster_mapping = {}
    for cluster_num in range(5):  
        most_common_label = true_labels[cluster_labels == cluster_num].mode()[0]
        cluster_mapping[cluster_num] = most_common_label

    data['predicted_labels'] = cluster_labels.map(cluster_mapping)
    
    precision = precision_score(true_labels, data['predicted_labels'], average='weighted', zero_division=1)
    recall = recall_score(true_labels, data['predicted_labels'], average='weighted', zero_division=1)
    f1 = f1_score(true_labels, data['predicted_labels'], average='weighted', zero_division=1)
    accuracy = accuracy_score(true_labels, data['predicted_labels'])

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    
 # print("\nClassification Report:\n", classification_report(true_labels, data['predicted_labels'], target_names=le.classes_))
else:
    print("The dataset must contain 'genre' and 'popularity' columns.")
