import pandas as pd
import os


def concatenate_and_clean_csvs():
    # Read the CSV files
    print("Reading CSV files...")
    base_dir = "/home/ubuntu/new_code"
    final_cleaned_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/final_cleaned.csv")
    png_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/data_with_png_paths.csv")

    df_cleaned = pd.read_csv(final_cleaned_path)
    df_png_paths = pd.read_csv(png_path)

    # Display initial information
    print("\nInitial shapes:")
    print(f"final_cleaned.csv: {df_cleaned.shape}")
    print(f"data_with_png_paths.csv: {df_png_paths.shape}")

    # Clean up first dataframe
    df_cleaned = df_cleaned.drop(columns=['Unnamed: 0'])
    df_cleaned = df_cleaned.rename(columns={
        'findings': 'findings',
        'Report Impression': 'impression'
    })

    # Prepare second dataframe
    # Convert dicom_id to png format and use it as image_id
    df_png_paths['image_id'] = df_png_paths['dicom_id'].str.replace('.dcm', '.png')
    df_png_paths = df_png_paths.rename(columns={
        'findings': 'findings',
        'impressions': 'impression'
    })

    # Select and reorder columns for second dataframe to match first
    df_png_paths = df_png_paths[[
        'image_id',
        'dicom_path',  # This will be our image_path for the second dataset
        'findings',
        'impression'
    ]]

    # Rename dicom_path to image_path for consistency
    df_png_paths = df_png_paths.rename(columns={'dicom_path': 'image_path'})

    # Concatenate the dataframes
    print("\nConcatenating dataframes...")
    combined_df = pd.concat([df_cleaned, df_png_paths], ignore_index=True)

    # Display final information
    print("\nFinal shape:", combined_df.shape)
    print("\nColumns in combined dataset:", combined_df.columns.tolist())

    # Check for any missing values
    print("\nMissing values in combined dataset:")
    print(combined_df.isnull().sum())

    # Check for duplicate image_ids
    duplicates = combined_df['image_id'].duplicated().sum()
    print(f"\nNumber of duplicate image_ids: {duplicates}")

    # Save the combined dataframe
    output_filename = 'combined_findings_2.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"\nSaved combined dataset to {output_filename}")

    return combined_df


if __name__ == "__main__":
    try:
        combined_df = concatenate_and_clean_csvs()

        # Display a sample of the combined data
        print("\nSample of combined dataset:")
        print(combined_df.head())

    except FileNotFoundError as e:
        print(f"Error: Could not find one of the CSV files. Please ensure both files are in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")