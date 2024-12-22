import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import naive_bayes as NB


precision, recall, f1, support = precision_recall_fscore_support(NB.y_test, NB.y_pred, labels=NB.gnb.classes_)

# DataFrame for visualization
metrics_df = NB.pd.DataFrame({
    'Class': NB.gnb.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Plot Precision, Recall, and F1-Score
metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-Score']].plot(kind='bar', figsize=(12, 6))
plt.title('Class-wise Precision, Recall, and F1-Score')
plt.ylabel('Score')
plt.xlabel('Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Support Distribution
metrics_df.set_index('Class')['Support'].plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title('Class-wise Support (Number of Samples)')
plt.ylabel('Number of Samples')
plt.xlabel('Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap of the Confusion Matrix
plt.figure(figsize=(10, 8))
cm = NB.confusion_matrix(NB.y_test, NB.y_pred, labels=NB.gnb.classes_)

sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=NB.gnb.classes_, yticklabels=NB.gnb.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
