import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, matthews_corrcoef)

from utils.save_data import create_folders


class ModelTrainer:
    def __init__(self, root_path: str, models_path: str, evidences_path: str):
        self.root_path = root_path
        self.dataframes_features_folder = f"{root_path}/dataframes"
        self.models_path = models_path
        self.evidences_path = evidences_path

        self.models_names = {
            "logistic_regression": "Logistic Regression",
            "svm": "SVM",
            "decision_tree": "Decision Tree",
            "lda": "LDA",
            "1nn": "k-NN (k = 1)",
            "3nn": "k-NN (k = 3)",
            "ensemble": "Ensemble"
        }

        self.models_metrics = {
            "Key-Points Extractor": [],
            "Dataset": [],
            "Model": [],
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "F1-Score": [],
            "MCC": []
        }

    def load_data(self, folder: str, feature_file: str, dataset: str | None = None):
        """
        Load train and test data.

        :param folder: str
        :param feature_file: str
        :param dataset: str | None
        :return:
        """

        if dataset:
            train_dir = os.path.join(self.dataframes_features_folder, folder, dataset, "treino")
            test_dir = os.path.join(self.dataframes_features_folder, folder, dataset, "teste")
        else:
            train_dir = os.path.join(self.dataframes_features_folder, folder, "treino")
            test_dir = os.path.join(self.dataframes_features_folder, folder, "teste")

        train_df = pd.read_csv(os.path.join(train_dir, feature_file))
        X_train_paths = train_df["feature_path"].values
        X_train = [np.load(file).flatten() for file in X_train_paths]
        y_train = train_df["label"].values

        test_df = pd.read_csv(os.path.join(test_dir, feature_file))
        X_test_paths = test_df["feature_path"].values
        X_test = [np.load(file).flatten() for file in X_test_paths]
        y_test = test_df["label"].values

        return X_train, y_train, X_test, y_test

    def _save_confusion_matrix(self, y_test, y_pred, model_name: str, folder: str, feature_name: str,
                               dataset: str | None = None):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {feature_name} - {model_name}")

        evidences_dir = os.path.join(self.evidences_path, folder, dataset if dataset else '', feature_name)
        create_folders(evidences_dir)

        cm_filename = os.path.join(evidences_dir, f"cm_{model_name}.png")
        plt.savefig(cm_filename)
        plt.close()
        print(">> Confusion Matrix saved.")

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, feature_name: str, folder: str,
                           dataset: str | None = None):
        loo = LeaveOneOut()
        models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "svm": SVC(kernel="linear", random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "lda": LinearDiscriminantAnalysis(),
            "1nn": KNeighborsClassifier(n_neighbors=1),
            "3nn": KNeighborsClassifier(n_neighbors=3),
            "ensemble": VotingClassifier(estimators=[
                ('lr', LogisticRegression(random_state=42)),
                ('svm', SVC(kernel='linear', probability=True, random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('lda', LinearDiscriminantAnalysis()),
                ('knn1', KNeighborsClassifier(n_neighbors=1)),
                ('knn3', KNeighborsClassifier(n_neighbors=3))
            ], voting='soft')
        }

        for model_name_, model in models.items():
            try:
                model_name = self.models_names[model_name_]

                scores = cross_val_score(model, X_train, y_train, cv=loo)
                print(f"\n>> {model_name} {feature_name} LOSO Cross-Validation Accuracy: {scores.mean()}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)

                self.models_metrics["Key-Points Extractor"].append(folder)
                self.models_metrics["Dataset"].append(dataset if dataset else '-')
                self.models_metrics["Model"].append(model_name)
                self.models_metrics["Accuracy"].append(accuracy)
                self.models_metrics["Precision"].append(precision)
                self.models_metrics["Recall"].append(recall)
                self.models_metrics["F1-Score"].append(f1)
                self.models_metrics["MCC"].append(mcc)

                print(
                    f">> {model_name} Metrics\n"
                    f"   Accuracy: {accuracy}\n"
                    f"   Precision: {precision}\n"
                    f"   Recall: {recall}\n"
                    f"   F1-Score: {f1}\n"
                    f"   MCC: {mcc}"
                )

                models_dir = os.path.join(self.models_path, folder, dataset if dataset else '', feature_name)
                create_folders(models_dir)
                model_filename = os.path.join(models_dir, f"{model_name_}.pkl")
                joblib.dump(model, model_filename)

                self._save_confusion_matrix(y_test, y_pred, model_name, folder, feature_name, dataset)
            except Exception as e:
                print(f"Problem with {model_name_} training")
                print(e)

    def save_metrics(self):
        df = pd.DataFrame(self.models_metrics)
        df_path = os.path.join(self.evidences_path, "models_metrics.csv")
        df.to_csv(df_path, index=False)
