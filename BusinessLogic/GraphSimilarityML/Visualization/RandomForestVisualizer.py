import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc


class RandomForestVisualizer:
    def __init__(self, checkbox_values, rf_uuid):
        self.checkbox_values = checkbox_values
        self.saved_visualisation_dir = "MachineLearningData/visualization"
        self.rf_uuid = rf_uuid

        self.save_path = os.path.join(self.saved_visualisation_dir,
                                      "rf_" + str(rf_uuid))
        os.makedirs(self.save_path, exist_ok=True)

    def visualize_test(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Similar', 'Similar'],
                    yticklabels=['Not Similar', 'Similar'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        save_path = os.path.join(self.save_path, 'rf_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

    def visualize_feature_importance(self,
                                     feature_importances,
                                     n_graphlets=30):

        feature_labels = []
        for i in range(n_graphlets):
            feature_labels.append(f"Graph1_Graphlet{i + 1}")
        for i in range(n_graphlets):
            feature_labels.append(f"Graph2_Graphlet{i + 1}")

        importance_df = pd.DataFrame({
            'Feature': feature_labels,
            'Importance': feature_importances
        })

        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title('Top 15 Most Important Graphlets for Similarity Prediction')
        plt.tight_layout()

        save_path = os.path.join(self.save_path, 'rf_feature_importance.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

    def visualize_roc_curve(self, y_test, y_prob):
        y_prob = y_prob[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_path, 'rf_roc_curve.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

    def visualize_classification_report(self, y_test, y_pred):
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        df = pd.DataFrame(report_dict).T

        df = df.drop(columns=["support"], errors="ignore")

        plt.figure(figsize=(8, 5))
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Classification Report Heatmap")

        save_path = os.path.join(self.save_path, 'rf_classification_report.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

    def visualize_based_on_checkbox(self, y_test, y_pred, y_prob, feature_importances, n_graphlets=30):
        if self.checkbox_values["important_features"]:
            self.visualize_feature_importance(feature_importances, n_graphlets)
        if self.checkbox_values["confusion_matrix"]:
            self.visualize_test(y_test, y_pred)
        if self.checkbox_values["roc_curve"]:
            self.visualize_roc_curve(y_test, y_prob)
        if self.checkbox_values["classification_report"]:
            self.visualize_classification_report(y_test, y_pred)