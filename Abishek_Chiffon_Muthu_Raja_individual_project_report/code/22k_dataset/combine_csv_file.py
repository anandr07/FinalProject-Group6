import pandas as pd
import os

import pandas as pd

import pandas as pd
import numpy as np


def combine_csv_files(file1_path, file2_path, output_path):
    """
    Combines two CSV files and modifies the image_id column for specified rows,
    ensuring proper string handling of image paths.

    Parameters:
    file1_path (str): Path to first CSV file
    file2_path (str): Path to second CSV file
    output_path (str): Path where the combined CSV will be saved
    """
    # Read the CSV files with string type for image_id column
    df1 = pd.read_csv(file1_path, dtype={'image_id': str})
    df2 = pd.read_csv(file2_path, dtype={'image_id': str})

    # Replace NaN values in image_id with empty string
    df1['image_id'] = df1['image_id'].fillna('')
    df2['image_id'] = df2['image_id'].fillna('')

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Remove any rows where image_id is empty
    combined_df = combined_df[combined_df['image_id'] != '']

    # Path to append
    path_to_append = "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images/"

    # Modify image_id column for first 7470 rows, only if they exist and have valid image_id
    num_rows_to_modify = min(7470, len(combined_df))
    combined_df.loc[:num_rows_to_modify - 1, 'image_id'] = (
            path_to_append + combined_df.loc[:num_rows_to_modify - 1, 'image_id'].astype(str)
    )

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_path, index=False)

    # Print information about the process
    print(f"Combined {len(df1)} rows from first file and {len(df2)} rows from second file")
    print(f"Total rows in combined file after cleaning: {len(combined_df)}")
    print(f"Modified image_id path for first {num_rows_to_modify} rows")

    # Validate the data
    print("\nValidation:")
    print(f"Number of empty image_ids: {combined_df['image_id'].eq('').sum()}")
    print(f"Sample of first few image_ids:")
    print(combined_df['image_id'].head())
#
# #Example usage
# if __name__ == "__main__":
#     # Replace these with your actual file paths
#     base_dir = "/home/ubuntu/new_code"
#     file1_path = os.path.join(base_dir, "Data/final_cleaned.csv")
#     file2_path = os.path.join(base_dir, "data2/data_with_png_paths.csv")
#     output_path = os.path.join(base_dir, "other_models/22k_dataset/complete_findings.csv")
#
#     combine_csv_files(file1_path, file2_path, output_path)


# # Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    base_dir = "/home/ubuntu/new_code"
    file1_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/labeled_reports_with_images_7k.csv")
    file2_path = os.path.join(base_dir, "other_models/22k_dataset/old_data/labeled_reports_with_images_8k.csv")
    output_path = os.path.join(base_dir, "other_models/22k_dataset/complete_labels_2.csv")

    combine_csv_files(file1_path, file2_path, output_path)