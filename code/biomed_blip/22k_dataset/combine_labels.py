import pandas as pd
import os


def concatenate_label_files():
    # Read the CSV files
    print("Reading CSV files...")

    base_dir = "/home/ubuntu/new_code"
    label_7k_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/labeled_reports_with_images_7k.csv")
    label_8k_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/labeled_reports_with_images_8k.csv")

    labels_7k = pd.read_csv(label_7k_path)
    labels_8k = pd.read_csv(label_8k_path)

    # Display initial information
    print("\nInitial shapes:")
    print(f"labels_with_7k_image.csv: {labels_7k.shape}")
    print(f"labels_with_8k_images.csv: {labels_8k.shape}")

    # Convert .dcm to .png in the 8k dataset
    print("\nConverting .dcm to .png in 8k dataset image_ids...")
    labels_8k['image_id'] = labels_8k['image_id'].str.replace('.dcm', '.png')

    # Verify columns match
    print("\nChecking columns in both datasets...")
    print("7k columns:", labels_7k.columns.tolist())
    print("8k columns:", labels_8k.columns.tolist())

    # Ensure column order matches exactly
    expected_columns = [
        'image_id', 'Report Impression', 'Enlarged Cardiomediastinum',
        'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
        'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
        'Pleural Effusion', 'Pleural Other', 'Fracture',
        'Support Devices', 'No Finding'
    ]

    # Reorder columns in both dataframes
    labels_7k = labels_7k[expected_columns]
    labels_8k = labels_8k[expected_columns]

    # Concatenate the dataframes
    print("\nConcatenating dataframes...")
    combined_df = pd.concat([labels_7k, labels_8k], ignore_index=True)

    # Display final information
    print("\nFinal shape:", combined_df.shape)
    print(f"Total rows: {len(combined_df)} (should be close to 15k)")

    # Check for any missing values
    print("\nMissing values in combined dataset:")
    print(combined_df.isnull().sum())

    # Check for duplicate image_ids
    duplicates = combined_df['image_id'].duplicated().sum()
    print(f"\nNumber of duplicate image_ids: {duplicates}")

    # Basic statistics for label columns
    print("\nLabel distribution in combined dataset:")
    label_columns = [col for col in combined_df.columns if col not in ['image_id', 'Report Impression']]
    label_stats = combined_df[label_columns].sum().sort_values(ascending=False)
    print(label_stats)

    # Save the combined dataframe
    output_filename = 'combined_labels_2.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"\nSaved combined dataset to {output_filename}")

    return combined_df


if __name__ == "__main__":
    try:
        combined_df = concatenate_label_files()

        # Display a sample of the combined data
        print("\nSample of combined dataset:")
        print(combined_df.head())

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files. Please ensure both files are in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")