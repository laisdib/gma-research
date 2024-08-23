import os

from steps.keypoints_preprocessing.preprocessing_pipeline import preprocessing_pipeline
from steps.keypoints_normalizing.normalization_pipeline import normalization_pipeline
from steps.features_extraction.features_extraction_pipeline import features_extraction_pipeline
from steps.features_standardization.standardization_pipeline import standardization_pipeline
from steps.features_fusion.fusion_pipeline import fusion_pipeline
from steps.df_generation.df_generation_pipeline import df_generation_pipeline
from steps.models_training.models_training_pipeline import models_training_pipeline


root_path = os.path.join(os.getcwd(), "data", "key-points")
original_key_points_folder = os.path.join(root_path, "original")
normalized_key_points_folder = os.path.join(root_path, "normalized")

print(">> Preprocessing key-points...")
preprocessing_pipeline(original_key_points_folder)

print(">> Normalizing key-points...")
normalization_pipeline(original_key_points_folder)

print(">> Extracting features...")
features_extraction_pipeline(normalized_key_points_folder)

root_path = os.path.join(os.getcwd(), "data", "features")
non_standardized_features_folder = os.path.join(root_path, "non_standardized")
standardized_features_folder = os.path.join(root_path, "standardized")
fused_features_folder = os.path.join(root_path, "fused")

unit_features = ["FFT-JD", "FFT-JO", "HOAD2D", "HOJD2D", "HOJO2D", "HORJAD2D", "HORJO2D"]
fused_features = ["pose_based", "velocity_based", "all_features"]

print(">> Standardizing features...")
standardization_pipeline(non_standardized_features_folder)

print(">> Fusing features...")
fusion_pipeline(standardized_features_folder)

print(">> Generating features dataframe...")
df_generation_pipeline(standardized_features_folder, fused_features_folder, unit_features, fused_features)

models_path = os.path.join(os.getcwd(), "evidences", "trained_models")
evidences_path = os.path.join(os.getcwd(), "evidences", "evaluation_artifacts")
dataframes_features_folder = os.path.join(root_path, "dataframes")

print(">> Training models...")
models_training_pipeline(root_path, models_path, evidences_path, dataframes_features_folder)

print("\n>> GMA pipeline done!")
