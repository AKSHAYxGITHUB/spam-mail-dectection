import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile

class DatasetLoader:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.main_dataset_path = os.path.join(data_dir, 'spam.csv')

    def load_local_spam_csv(self):
        """Loads the default spam.csv which is typically the SMS Spam Collection."""
        print("Loading local spam.csv...")
        try:
            # spam.csv usually has encoding issues, so we try latin-1
            df = pd.read_csv(self.main_dataset_path, encoding='latin-1')
            # Select relevant columns and rename universally
            df = df[['v1', 'v2']]
            df.columns = ['label', 'message']
            # Convert label to binary: 1 for spam, 0 for ham
            df['label'] = df['label'].astype(str).str.lower().str.strip()
            df['label'] = df['label'].map({'spam': 1, 'ham': 0, '1': 1, '0': 0})
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            return df
        except Exception as e:
            print(f"Error loading local spam.csv: {e}")
            return pd.DataFrame(columns=['label', 'message'])

    def fetch_additional_datasets(self):
        """
        Placeholder for fetching Enron, Ling Spam, etc.
        In a full production environment, this would call APIs or download zip files.
        For now, we return an empty dataframe, but structure is here.
        """
        # Example structure
        # df_enron = self.load_enron()
        return pd.DataFrame(columns=['label', 'message'])

    def get_combined_dataset(self):
        """Loads and merges all datasets, handles duplicates."""
        df_local = self.load_local_spam_csv()
        df_additional = self.fetch_additional_datasets()
        
        # Merge
        df_combined = pd.concat([df_local, df_additional], ignore_index=True)
        
        # Clean: remove duplicates
        initial_len = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['message'])
        final_len = len(df_combined)
        print(f"Removed {initial_len - final_len} duplicate messages.")
        
        # Clean: remove nulls or empty strings
        df_combined['message'] = df_combined['message'].fillna('')
        df_combined = df_combined[df_combined['message'].str.strip() != '']
        
        # Subsample if dataset is too large (to prevent SVM from hanging/OOM)
        MAX_ROWS = 40000
        if len(df_combined) > MAX_ROWS:
            print(f"Dataset is huge ({len(df_combined)} rows). Subsampling to {MAX_ROWS} rows for memory-safe training...")
            # We must handle NaN labels before stratifying
            df_combined = df_combined.dropna(subset=['label'])
            from sklearn.model_selection import train_test_split
            df_combined, _ = train_test_split(df_combined, train_size=MAX_ROWS, stratify=df_combined['label'], random_state=42)
        
        print(f"Total dataset size after merging and cleaning: {len(df_combined)}")
        return df_combined

if __name__ == '__main__':
    loader = DatasetLoader()
    df = loader.get_combined_dataset()
    print(df.head())
    print(df['label'].value_counts())
