import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CoswaraDataMerger:
    def __init__(self, metadata_csv_path, audio_base_path):
        self.metadata_csv_path = metadata_csv_path
        self.audio_base_path = audio_base_path
        self.metadata_df = None
        self.audio_paths_df = None
        
    def load_metadata(self):
        """Load and clean the metadata CSV"""
        print("Loading metadata CSV...")
        self.metadata_df = pd.read_csv(self.metadata_csv_path)
        
        # Display basic info
        print(f"Metadata shape: {self.metadata_df.shape}")
        print(f"Columns: {list(self.metadata_df.columns)}")
        if 'covid_status' in self.metadata_df.columns:
            print(f"COVID status distribution:")
            print(self.metadata_df['covid_status'].value_counts())
        
        # Show first few patient IDs to verify
        if 'id' in self.metadata_df.columns:
            print(f"Sample patient IDs: {self.metadata_df['id'].head(5).tolist()}")
        
        return self.metadata_df
    
    def find_audio_files(self):
        """Find all audio files and map them to patient IDs"""
        print("Scanning for audio files...")
        
        audio_data = []
        
        # Get all date folders in Extracted_data
        date_folders = [f for f in os.listdir(self.audio_base_path) 
                       if os.path.isdir(os.path.join(self.audio_base_path, f))]
        print(f"Found {len(date_folders)} date folders")
        
        for date_folder in tqdm(date_folders):
            date_path = os.path.join(self.audio_base_path, date_folder)
            
            # Get all patient folders for this date
            patient_folders = [f for f in os.listdir(date_path) 
                             if os.path.isdir(os.path.join(date_path, f))]
            
            for patient_id in patient_folders:
                patient_path = os.path.join(date_path, patient_id)
                
                # Check for audio files
                audio_files = {}
                expected_audio_types = [
                    'breathing-deep.wav', 'breathing-shallow.wav',
                    'cough-heavy.wav', 'cough-shallow.wav',
                    'counting-normal.wav', 'counting-fast.wav',
                    'vowel-a.wav', 'vowel-e.wav', 'vowel-o.wav'
                ]
                
                for audio_type in expected_audio_types:
                    audio_path = os.path.join(patient_path, audio_type)
                    if os.path.exists(audio_path):
                        # Store relative path for portability
                        audio_files[audio_type.replace('.wav', '').replace('-', '_')] = audio_path
                        audio_files[f'has_{audio_type.replace(".wav", "").replace("-", "_")}'] = 1
                    else:
                        audio_files[f'has_{audio_type.replace(".wav", "").replace("-", "_")}'] = 0
                
                # Add to collection
                if any([audio_files.get(f'has_{audio_type.replace(".wav", "").replace("-", "_")}', 0) 
                       for audio_type in expected_audio_types]):
                    audio_record = {
                        'patient_id': patient_id,
                        'date_folder': date_folder,
                        **audio_files
                    }
                    audio_data.append(audio_record)
        
        self.audio_paths_df = pd.DataFrame(audio_data)
        print(f"Found {len(self.audio_paths_df)} patients with audio files")
        
        # Show sample of patient IDs found
        print(f"Sample audio patient IDs: {self.audio_paths_df['patient_id'].head(5).tolist()}")
        
        return self.audio_paths_df
    
    def merge_datasets(self):
        """Merge metadata with audio file paths"""
        if self.metadata_df is None:
            self.load_metadata()
        if self.audio_paths_df is None:
            self.find_audio_files()
        
        print("Merging datasets...")
        
        # Try different possible patient ID columns
        patient_id_columns = ['id', 'd', 'patient_id']  # Try these in order
        
        left_column = None
        for col in patient_id_columns:
            if col in self.metadata_df.columns:
                left_column = col
                print(f"Using '{col}' as patient ID column from metadata")
                break
        
        if left_column is None:
            print("Error: Could not find patient ID column in metadata.")
            print(f"Available columns: {list(self.metadata_df.columns)}")
            return None
        
        # Merge on patient ID
        merged_df = pd.merge(
            self.metadata_df, 
            self.audio_paths_df, 
            left_on=left_column,  # Patient ID column in metadata
            right_on='patient_id', 
            how='inner'
        )
        
        print(f"Merged dataset shape: {merged_df.shape}")
        if 'covid_status' in merged_df.columns:
            print(f"COVID status in merged data:")
            print(merged_df['covid_status'].value_counts())
        
        return merged_df
    
    def create_final_dataset(self, output_path=None):
        """Create the final merged dataset with proper target variable"""
        merged_df = self.merge_datasets()
        
        if merged_df is None or merged_df.empty:
            print("Error: No data to process")
            return None
        
        # Create binary target variable - map COVID status to binary
        if 'covid_status' in merged_df.columns:
            print("Creating target variable...")
            
            # Map COVID status to binary
            covid_mapping = {
                'positive_mild': 1,
                'positive_moderate': 1,
                'positive_asymp': 1,
                'healthy': 0,
                'no_resp_illness_exposed': 0,
                'recovered_full': 0
            }
            
            merged_df['target'] = merged_df['covid_status'].map(covid_mapping)
            
            # Count before filtering
            print(f"Records before filtering unknown status: {len(merged_df)}")
            
            # Handle any unknown COVID status (remove under_validation and resp_illness_not_identified)
            merged_df = merged_df[merged_df['target'].notna()]
            
            print(f"Records after filtering: {len(merged_df)}")
            print(f"Target distribution:")
            print(merged_df['target'].value_counts())
            print(f"COVID positive: {merged_df['target'].sum()}")
            print(f"COVID negative: {len(merged_df) - merged_df['target'].sum()}")
        else:
            print("Warning: 'covid_status' column not found. Target variable not created.")
            merged_df['target'] = 0  # Default to negative
        
        # Add useful derived features
        audio_has_columns = [col for col in merged_df.columns if col.startswith('has_')]
        if audio_has_columns:
            merged_df['has_all_audio'] = merged_df[audio_has_columns].all(axis=1)
            merged_df['num_audio_types'] = merged_df[audio_has_columns].sum(axis=1)
            print(f"Average audio types per patient: {merged_df['num_audio_types'].mean():.2f}")
        
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            merged_df.to_csv(output_path, index=False)
            print(f"Final dataset saved to: {output_path}")
        
        return merged_df