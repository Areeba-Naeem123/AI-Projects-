import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
from joblib import dump


# ================================ Load and Clean Data ================================
data = pd.read_csv('dataset.csv')
data_cleaned = data.dropna().drop_duplicates()
data_cleaned = data_cleaned[data_cleaned['popularity'] != 0]

data_cleaned['explicit_binary'] = data_cleaned['explicit'].astype(int)
data_cleaned['danceability_energy'] = data_cleaned['danceability'] * data_cleaned['energy']
data_cleaned['duration_log'] = np.log1p(data_cleaned['duration_ms'])
data_cleaned['acoustic_energy_ratio'] = data_cleaned['acousticness'] / (data_cleaned['energy'] + 1e-6)

# ================================ Genre Mapping ================================
genre_mapping = {
    'pop': 'Pop/Dance', 'power-pop': 'Pop/Dance', 'dance': 'Pop/Dance', 'club': 'Pop/Dance',
    'rock': 'Rock', 'alt-rock': 'Rock', 'hard-rock': 'Rock', 'metal': 'Rock',
    'hip-hop': 'Hip-Hop/Rap', 'rap': 'Hip-Hop/Rap', 'trap': 'Hip-Hop/Rap',
    'classical': 'Classical/Instrumental', 'instrumental': 'Classical/Instrumental',
    'edm': 'Electronic', 'house': 'Electronic', 'techno': 'Electronic',
    'jazz': 'Other-Misc', 'blues': 'Other-Misc', 'latin': 'Other-Misc', 'folk': 'Other-Misc'
}
data_cleaned['broad_genre'] = data_cleaned['track_genre'].map(genre_mapping).fillna('Other-Misc')

# ================================ Downsample "Other-Misc" ================================
target_count = 10000
other_misc = data_cleaned[data_cleaned['broad_genre'] == 'Other-Misc'].sample(n=target_count, random_state=42)
remaining_data = data_cleaned[data_cleaned['broad_genre'] != 'Other-Misc']
data_balanced = pd.concat([remaining_data, other_misc])

print("Class Distribution After Balancing:")
print(data_balanced['broad_genre'].value_counts())

# ================================ Features and Target ================================
features = ['popularity', 'duration_log', 'danceability', 'energy', 'acousticness', 
            'danceability_energy', 'acoustic_energy_ratio']
X = data_balanced[features]
y = data_balanced['broad_genre']

# SMOTE-Tomek for Balancing
smote_tomek = SMOTETomek(random_state=57)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# ================================ Feature Transformation ================================
scaler = StandardScaler()
quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = quantile_transformer.fit_transform(scaler.fit_transform(X_train))
X_test_scaled = quantile_transformer.transform(scaler.transform(X_test))

# ================================ Train Ensemble Classifier ================================
gnb = GaussianNB()
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

ensemble_clf = VotingClassifier(estimators=[
    ('gnb', gnb), ('lr', lr), ('rf', rf)
], voting='soft', weights=[1, 2, 3])

ensemble_clf.fit(X_train_scaled, y_train)
y_pred = ensemble_clf.predict(X_test_scaled)

rf_from_ensemble = ensemble_clf.named_estimators_['rf']
dump(rf_from_ensemble, 'random_forest_model.pkl')
# ================================ Post-Prediction Threshold Adjustment ================================
from sklearn.preprocessing import label_binarize
y_proba = ensemble_clf.predict_proba(X_test_scaled)

thresholds = {'Other-Misc': 0.3, 'Pop/Dance': 0.3}
y_pred_adjusted = []

for i, probs in enumerate(y_proba):
    max_class = ensemble_clf.classes_[np.argmax(probs)]
    for cls, threshold in thresholds.items():
        cls_idx = list(ensemble_clf.classes_).index(cls)
        if probs[cls_idx] > threshold:
            max_class = cls
    y_pred_adjusted.append(max_class)

# ================================ Model Evaluation ================================
print("\nClassification Report (Final Model):")
print(classification_report(y_test, y_pred_adjusted))

cm = confusion_matrix(y_test, y_pred_adjusted, labels=ensemble_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_clf.classes_)
disp.plot(cmap='viridis', xticks_rotation=90)
plt.title("Final Model Performance")
plt.show() 
