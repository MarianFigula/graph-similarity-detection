import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from PIL import Image


class MLPVisualizer:
    def __init__(self, checkbox_values, mlp_uuid):
        self.checkbox_values = checkbox_values
        self.saved_visualisation_dir = "MachineLearningData/visualization"
        self.mlp_uuid = mlp_uuid

        self.save_path = os.path.join(self.saved_visualisation_dir,
                                      "mlp_" + str(mlp_uuid))
        os.makedirs(self.save_path, exist_ok=True)

    # accuracy and loss plot
    def visualize_training_history(self, history):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='lower right')
        ax1.grid(True)

        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper right')
        ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.save_path, 'mlp_training_history.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

        # plt.show()

    def visualize_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix (Test Data)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join(self.save_path, 'mlp_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()
        # plt.show()

    def visualize_roc_curve(self, y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        save_path = os.path.join(self.save_path, 'mlp_roc_curve.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

        # plt.show()

    def visualize_classification_report(self, y_test, y_pred):
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        df = pd.DataFrame(report_dict).T

        df = df.drop(columns=["support"], errors="ignore")

        # Plot heatmap
        plt.figure(figsize=(8, 5))
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Classification Report Heatmap")

        save_path = os.path.join(self.save_path, 'mlp_classification_report.png')
        plt.savefig(save_path)
        plt.close()

        img = Image.open(save_path)
        img.show()

        # plt.show()

    def visualize_based_on_checkbox(self, history, y_test, y_pred, y_prob):
        if self.checkbox_values["accuracy_loss"]:
            self.visualize_training_history(history)
        if self.checkbox_values["confusion_matrix"]:
            self.visualize_confusion_matrix(y_test, y_pred)
        if self.checkbox_values["roc_curve"]:
            self.visualize_roc_curve(y_test, y_prob)
        if self.checkbox_values["classification_report"]:
            self.visualize_classification_report(y_test, y_pred)
