import pandas as pd

def inspect_and_merge():
    # 1. Read main spam.csv
    try:
        df_main = pd.read_csv('spam.csv', encoding='latin-1')
        print(f"Main spam.csv shape: {df_main.shape}")
        print(f"Main columns: {list(df_main.columns)}")
    except Exception as e:
        print(f"Error reading main spam.csv: {e}")
        df_main = pd.DataFrame(columns=['v1', 'v2'])

    # 2. Read email_spam_dataset.csv
    try:
        df1 = pd.read_csv('email_spam_dataset.csv', encoding='latin-1')
        print(f"Dataset 1 shape: {df1.shape}")
        print(f"Dataset 1 columns: {list(df1.columns)}")
        # Assuming typical columns like 'label', 'text' or 'v1', 'v2'
        if 'label' in df1.columns and 'text' in df1.columns:
            df1 = df1.rename(columns={'label': 'v1', 'text': 'v2'})
        elif 'Category' in df1.columns and 'Message' in df1.columns:
            df1 = df1.rename(columns={'Category': 'v1', 'Message': 'v2'})
            
        # Keep only relevant columns
        if 'v1' in df1.columns and 'v2' in df1.columns:
            df1 = df1[['v1', 'v2']]
            # Standardize labels
            df1['v1'] = df1['v1'].astype(str).str.lower().str.strip()
            df1['v1'] = df1['v1'].map({'spam': 'spam', 'ham': 'ham', '1': 'spam', '0': 'ham'})
        else:
            df1 = pd.DataFrame(columns=['v1', 'v2'])
    except Exception as e:
        print(f"Error reading email_spam_dataset.csv: {e}")
        df1 = pd.DataFrame(columns=['v1', 'v2'])

    # 3. Read spam_Emails_data.csv
    try:
        # Read a sample first to determine columns safely
        sample = pd.read_csv('spam_Emails_data.csv', encoding='utf-8', on_bad_lines='skip', nrows=5)
        print(f"Dataset 2 sample columns: {list(sample.columns)}")
        
        # Mapping logic based on common spam dataset formats
        col_map = {}
        if 'text' in sample.columns and 'spam' in sample.columns:
            col_map = {'spam': 'v1', 'text': 'v2'}
        elif 'label' in sample.columns and 'text' in sample.columns:
            col_map = {'label': 'v1', 'text': 'v2'}
        elif 'Category' in sample.columns and 'Message' in sample.columns:
            col_map = {'Category': 'v1', 'Message': 'v2'}
        elif 'v1' in sample.columns and 'v2' in sample.columns:
            col_map = {'v1': 'v1', 'v2': 'v2'}
            
        # If we couldn't figure out columns from the sample
        if not col_map:
             print("Could not automatically determine columns for Dataset 2. Skipping.")
             df2 = pd.DataFrame(columns=['v1', 'v2'])
        else:
             print(f"Mapping for Dataset 2: {col_map}")
             # Read the full dataset
             df2 = pd.read_csv('spam_Emails_data.csv', encoding='utf-8', on_bad_lines='skip', usecols=list(col_map.keys()))
             df2 = df2.rename(columns=col_map)
             
             # Standardize labels
             df2['v1'] = df2['v1'].astype(str).str.lower().str.strip()
             df2['v1'] = df2['v1'].map({'spam': 'spam', 'ham': 'ham', '1': 'spam', '0': 'ham', '1.0': 'spam', '0.0': 'ham'})
             
             # Drop unmapped NAs to prevent messing up the dataset
             df2 = df2.dropna(subset=['v1', 'v2'])

    except Exception as e:
        print(f"Error reading spam_Emails_data.csv: {e}")
        df2 = pd.DataFrame(columns=['v1', 'v2'])

    # 4. Merge and deduplicate
    print("Merging datasets...")
    frames = [df_main, df1, df2]
    df_combined = pd.concat([df for df in frames if not df.empty], ignore_index=True)
    
    if not df_combined.empty:
        initial_len = len(df_combined)
        df_combined = df_combined.dropna(subset=['v1', 'v2'])
        df_combined = df_combined.drop_duplicates(subset=['v2'])
        final_len = len(df_combined)
        
        print(f"Combined shape before deduplication: {initial_len}")
        print(f"Removed {initial_len - final_len} duplicate/invalid records.")
        print(f"Final shape: {df_combined.shape}")
        
        # 5. Save back to spam.csv
        df_combined.to_csv('spam.csv', index=False, encoding='utf-8')
        print("Successfully updated spam.csv.")
    else:
        print("Final combined dataset is empty.")

if __name__ == "__main__":
    inspect_and_merge()
