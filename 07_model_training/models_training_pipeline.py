import os

from models_trainer import ModelTrainer
from utils.utils import define_dataset_path


def train_and_evaluate_models(folder_path_: str, model_trainer_: ModelTrainer, dataset_: str | None = None):
    """
    Train and evaluate models.

    :param folder_path_: str
    :param model_trainer_: ModelTrainer
    :param dataset_: str | None
    """

    dataset_path = define_dataset_path(folder_path_, dataset_)
    dataset_path = os.path.join(dataset_path, "treino")

    for feature_file_ in os.listdir(dataset_path):
        feature_name_ = feature_file_.replace(".csv", "")
        X_train, y_train, X_test, y_test = model_trainer_.load_data(folder, feature_file_, dataset_)
        model_trainer_.train_and_evaluate(X_train, y_train, X_test, y_test, feature_name_, folder, dataset_)


root_path = "../data/features"
dataframes_features_folder = f"{root_path}/dataframes"
models_path = "../trained_models"
evidences_path = "../evaluation_artifacts"

model_trainer = ModelTrainer(root_path, models_path, evidences_path)

for folder in os.listdir(dataframes_features_folder):
    folder_path = os.path.join(dataframes_features_folder, folder)
    datasets = os.listdir(folder_path)

    if "treino" in datasets:
        train_and_evaluate_models(folder_path, model_trainer)
    else:
        for dataset in datasets:
            train_and_evaluate_models(folder_path, model_trainer, dataset)

model_trainer.save_metrics()
