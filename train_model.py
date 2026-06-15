import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

from dataset_loader import DatasetLoader
from preprocessing import TextPreprocessor
from model_pipeline import ModelPipeline

def main():
    print("=== Spam Detection Training Pipeline ===")
    
    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    loader = DatasetLoader()
    df = loader.get_combined_dataset()
    
    if df.empty:
        print("Error: Dataset is empty.")
        return

    # 2. Preprocess Data
    print("\n[2/5] Preprocessing text data...")
    preprocessor = TextPreprocessor()
    df['cleaned_message'] = df['message'].apply(preprocessor.clean_text)
    
    X = df['cleaned_message'].values
    y = df['label'].values.astype(int)
    
    # Train-test split before synthetic augmentation/SMOTE to prevent data leakage
    X_train_raw, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Feature Engineering
    print("\n[3/5] Extracting features (TF-IDF)...")
    pipeline = ModelPipeline(representation='tfidf')
    X_train_vec, X_test_vec = pipeline.build_features(X_train_raw, X_test)

    # Dataset is already roughly balanced (~52% ham / 48% spam), so no
    # resampling is needed.
    print(f"Training Class Distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # 4. Train Model
    print("\n[4/5] Training ensemble model...")
    pipeline.build_ensemble_model()
    pipeline.train(X_train_vec, y_train)
    
    # 5. Evaluate and Save
    print("\n[5/5] Evaluating and Saving...")
    pipeline.evaluate(X_test_vec, y_test)
    pipeline.save_artifacts()
    
    # Export weights for zero-dependency inference on Vercel
    pipeline.save_models_to_json('model_params.json')
    pipeline.save_vectorizer_to_json('vectorizer.json')
    
    print("\n=== Pipeline Complete ===")

if __name__ == '__main__':
    main()