import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

# ================================ Load and Clean Data ================================
data = pd.read_csv('dataset.csv')
unique_entries = data['track_genre'].unique()
unique_count = len(unique_entries)
print(f"Number of unique entries: {unique_count}")
data_cleaned = data.dropna().drop_duplicates()
data_cleaned = data_cleaned[data_cleaned['popularity'] != 0]

data_cleaned['explicit_binary'] = data_cleaned['explicit'].astype(int)
data_cleaned['danceability_energy'] = data_cleaned['danceability'] * data_cleaned['energy']

# ================================ Genre Mapping ================================
genre_mapping = {
    'pop': 'Pop/Dance', 'power-pop': 'Pop/Dance', 'pop-film': 'Pop/Dance', 'indie-pop': 'Pop/Dance',
    'dance': 'Pop/Dance', 'club': 'Pop/Dance', 'party': 'Pop/Dance',
    'rock': 'Rock', 'alt-rock': 'Rock', 'punk-rock': 'Rock', 'hard-rock': 'Rock',
    'psych-rock': 'Rock', 'grunge': 'Rock', 'metal': 'Rock', 'heavy-metal': 'Rock',
    'punk': 'Rock', 'rockabilly': 'Rock',
    'edm': 'Electronic', 'house': 'Electronic', 'electro': 'Electronic', 'dubstep': 'Electronic',
    'techno': 'Electronic', 'minimal-techno': 'Electronic', 'progressive-house': 'Electronic',
    'detroit-techno': 'Electronic', 'deep-house': 'Electronic', 'chicago-house': 'Electronic',
    'hip-hop': 'Hip-Hop/Rap', 'rap': 'Hip-Hop/Rap', 'trap': 'Hip-Hop/Rap',
    'classical': 'Classical/Instrumental', 'opera': 'Classical/Instrumental',
    'piano': 'Classical/Instrumental', 'instrumental': 'Classical/Instrumental', 'ambient': 'Classical/Instrumental',
    # Merge low-performing classes
    'jazz': 'Other-Misc', 'blues': 'Other-Misc', 'bluegrass': 'Other-Misc', 'funk': 'Other-Misc',
    'soul': 'Other-Misc', 'gospel': 'Other-Misc', 'latin': 'Other-Misc', 'brazil': 'Other-Misc',
    'afrobeat': 'Other-Misc', 'salsa': 'Other-Misc', 'reggae': 'Other-Misc', 'folk': 'Other-Misc',
    'country': 'Other-Misc', 'anime': 'Other-Misc', 'children': 'Other-Misc', 'comedy': 'Other-Misc',
    'study': 'Other-Misc', 'happy': 'Other-Misc', 'sad': 'Other-Misc',
    'disney': 'Other-Misc', 'show-tunes': 'Other-Misc', 'songwriter': 'Other-Misc',
    'j-idol': 'Other-Misc', 'tango': 'Other-Misc', 'grindcore': 'Other-Misc', 'new-age': 'Other-Misc',
    'other': 'Other-Misc', 'other/miscellaneous': 'Other-Misc'
}

if 'track_genre' in data_cleaned.columns:
    data_cleaned['broad_genre'] = data_cleaned['track_genre'].map(genre_mapping).fillna('Other-Misc')
else:
    print("Error: Column 'track_genre' not found in the dataset.")
    exit()

# ================================ Data Balancing ================================
print("Class Distribution Before Balancing:")
print(data_cleaned['broad_genre'].value_counts())

target_count = 8000
other_misc = data_cleaned[data_cleaned['broad_genre'] == 'Other-Misc'].sample(n=target_count, random_state=42)
remaining_data = data_cleaned[data_cleaned['broad_genre'] != 'Other-Misc']
data_balanced = pd.concat([remaining_data, other_misc])

print("Class Distribution After Downsampling 'Other-Misc':")
print(data_balanced['broad_genre'].value_counts())

# ================================ Features and Target ================================
features = ['popularity', 'duration_ms', 'danceability', 'energy', 'acousticness']
X = data_balanced[features]
y = data_balanced['broad_genre']

# ================================ Feature Selection ================================
selector = SelectKBest(score_func=f_classif, k=4)
X_selected = selector.fit_transform(X, y)

# ================================ Handle Class Imbalance ================================
smote_tomek = SMOTETomek(random_state=51)
X_resampled, y_resampled = smote_tomek.fit_resample(X_selected, y)

# ================================ Train-Test Split ================================
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# ================================ Feature Transformation ================================
scaler = StandardScaler()
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)

X_train_scaled = quantile_transformer.fit_transform(scaler.fit_transform(X_train))
X_test_scaled = quantile_transformer.transform(scaler.transform(X_test))

# ================================ Train Gaussian Naive Bayes ================================
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# Predictions
y_pred = gnb.predict(X_test_scaled)

# ================================ Model Evaluation ================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

                                                                                                  