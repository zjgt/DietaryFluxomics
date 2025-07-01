import pandas as pd
import os
import glob
import numpy as np
import argparse
from collections import defaultdict


def process_csv_files(input_folder, output_folder):
    """
    Process multiple CSV files by grouping them based on filenames,
    extracting columns 5-10, renaming them, and combining them into
    separate output files for each group.
    Parameters:
    input_folder (str): Path to the folder containing CSV files
    output_folder (str): Path to the folder where output files will be saved
    """
    # Get all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    csv_files = np.sort(csv_files)  # Sort the files

    # Group files by their prefix (excluding the last three components)
    file_groups = defaultdict(list)
    for file_path in csv_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # Split filename by "_" and exclude the last three components
        #prefix = "_".join(file_name.split("_")[:-4])
        prefix = "_".join(file_name.split("_")[:-2])
        file_groups[prefix].append(file_path)

    # Process each group of files
    for prefix, files in file_groups.items():
        # List to store processed dataframes
        processed_dfs = []

        for file_path in files:
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Extract columns 5-15 (0-based indexing)
            selected_columns = df.iloc[:, 4:14]
            # Get the filename without extension for renaming
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            # Rename columns with filename_1 to filename_6
            new_column_names = [f"{file_name}_{i}" for i in range(1, 11)]
            selected_columns.columns = new_column_names
            # Add to list of processed dataframes
            processed_dfs.append(selected_columns)

        # Combine all processed dataframes
        if processed_dfs:
            final_df = pd.concat(processed_dfs, axis=1)
            # Create output subfolder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            # Define output file path
            output_file = os.path.join(output_folder, f"{prefix}_combined_data.csv")
            # Save to output file
            final_df.to_csv(output_file, index=False)
            print(f"Successfully processed {len(files)} files and saved to {output_file}")
        else:
            print(f"No CSV files found for group {prefix}")


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process CSV files and combine them based on filename prefixes.")
    parser.add_argument("input_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_folder", help="Path to the folder where output files will be saved")
    args = parser.parse_args()

    # Call the processing function
    process_csv_files(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()



# Example usage
#python Flux2_grid_search_result_summarize.py Flux2_Results Flux2_Results_Summary