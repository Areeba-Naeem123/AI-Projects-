
import combination_models as CV

from joblib import load
rf_from_ensemble = load('random_forest_model.pkl')

# Class Distribution After Balancing
CV.plt.figure(figsize=(12, 6))
CV.data_balanced['broad_genre'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
CV.plt.title('Class Distribution After Balancing', fontsize=16)
CV.plt.xlabel('Broad Genre', fontsize=14)
CV.plt.ylabel('Count', fontsize=14)
CV.plt.xticks(rotation=45, fontsize=12)
CV.plt.tight_layout()
CV.plt.show()


rf_importance = rf_from_ensemble.feature_importances_

# feature importances
CV.plt.figure(figsize=(10, 6))
CV.plt.barh(CV.features, rf_importance)
CV.plt.xlabel('Importance')
CV.plt.ylabel('Feature')
CV.plt.title('Feature Importance - RandomForestClassifier')
CV.plt.tight_layout()
CV.plt.show()

# Predicted Probabilities for Classes 
prob_df = CV.pd.DataFrame(CV.y_proba[:5], columns=CV.ensemble_clf.classes_)
prob_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
CV.plt.title('Predicted Probabilities for Top 5 Test Samples', fontsize=16)
CV.plt.xlabel('Sample Index', fontsize=14)
CV.plt.ylabel('Probability', fontsize=14)
CV.plt.xticks(rotation=0, fontsize=12)
CV.plt.tight_layout()
CV.plt.show()
