import sys
import os

# Add src to path so we can import from it
sys.path.append('src')

from data_merger import CoswaraDataMerger

def main():
    # Paths - these should work if you're in the Coswara-Data directory
    METADATA_CSV = "combined_data.csv"
    AUDIO_BASE_PATH = "Extracted_data"
    OUTPUT_PATH = "merged_coswara_dataset.csv"
    
    print("=== Coswara Data Merger ===")
    print(f"Metadata CSV: {METADATA_CSV}")
    print(f"Audio path: {AUDIO_BASE_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    
    # Check if files exist
    if not os.path.exists(METADATA_CSV):
        print(f"Error: {METADATA_CSV} not found!")
        return
    
    if not os.path.exists(AUDIO_BASE_PATH):
        print(f"Error: {AUDIO_BASE_PATH} not found!")
        return
    
    # Initialize and run merger
    merger = CoswaraDataMerger(METADATA_CSV, AUDIO_BASE_PATH)
    final_dataset = merger.create_final_dataset(OUTPUT_PATH)
    
    if final_dataset is not None:
        print("\n=== Process Complete ===")
        print(f"Final dataset created with {len(final_dataset)} patients")
        if 'target' in final_dataset.columns:
            covid_positive = final_dataset['target'].sum()
            covid_negative = len(final_dataset) - covid_positive
            print(f"COVID positive: {covid_positive}")
            print(f"COVID negative: {covid_negative}")
    else:
        print("\n=== Process Failed ===")

if __name__ == "__main__":
    main()