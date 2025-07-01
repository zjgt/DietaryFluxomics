import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import argparse

def replace_spaces_in_filenames(folder_path):

    for filename in os.listdir(folder_path):
        if " " in filename:
            new_filename = filename.replace(" ", "_")
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

def create_2d_contour_from_csvs(input_folder, output_folder, subfolder_name):
    # Lists to store coordinates and z values
    x_coords = []
    y_coords = []
    z_values = []
    # Iterate through CSV files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # Split filename into components
            components = filename.split('_')
            # Extract x and y coordinates from second to last and last components
            try:
                x = float(components[-2])
                y = float(components[-1].replace('.csv', ''))
                #x = float(components[-3])
                #y = float(components[-2])
                # Read the CSV file
                file_path = os.path.join(input_folder, filename)
                df = pd.read_csv(file_path)
                # Extract z value from the specified cell (5th column, 2nd row)
                # Convert to float and round up to integer
                z = math.ceil(float(df.iloc[0, 4]))
                # Append values
                x_coords.append(x)
                y_coords.append(y)
                z_values.append(z)
            except (ValueError, IndexError) as e:
                print(f"Skipping file {filename}: {e}")
    if not x_coords or not y_coords or not z_values:
        print(f"No valid data found in {input_folder}. Skipping...")
        return
    # Create grid for interpolation
    xi = np.linspace(min(x_coords), max(x_coords), 100)
    yi = np.linspace(min(y_coords), max(y_coords), 100)
    XI, YI = np.meshgrid(xi, yi)
    # Interpolate Z values
    ZI = griddata((x_coords, y_coords), z_values, (XI, YI), method='cubic')
    # Create 2D contour plot
    plt.figure(figsize=(10, 8))
    # Contour plot
    contour = plt.contourf(XI, YI, ZI, levels=50, cmap='viridis', alpha=0.75)
    # Scatter plot of original points
    plt.scatter(x_coords, y_coords, c=z_values, cmap='viridis', edgecolors='black', s=50)
    # Add color bar
    plt.colorbar(contour, label='RSS')
    # Labeling
    plt.xlabel('Glc_C13(x0.01)')
    plt.ylabel('Lac_C13(x0.01)')
    plt.title('2D Contour Plot')
    plt.tight_layout()
    # Save the plot instead of showing it
    output_path = os.path.join(output_folder, f'{subfolder_name}.png')
    plt.savefig(output_path)
    plt.close()

def sort_csv_files(input_folder):
    file_groups = {}
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            components = filename.split('_')[:-2]
            #components = filename.split('_')[:-4]
            group_key = '_'.join(components)
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(filename)
    return file_groups

def main():
    parser = argparse.ArgumentParser(description='Create 2D contour plots from CSV files.')
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing CSV files.')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder to save the plots.')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    replace_spaces_in_filenames(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_groups = sort_csv_files(input_folder)

    for group_key, filenames in file_groups.items():
        print(group_key)
        group_folder = os.path.join(output_folder, group_key)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)
        for filename in filenames:
            source_path = os.path.join(input_folder, filename)
            destination_path = os.path.join(group_folder, filename)
            os.rename(source_path, destination_path)
        create_2d_contour_from_csvs(group_folder, output_folder, group_key)

if __name__ == "__main__":
    main()

#python Flux2_grid_search_plot.py --input_folder Flux2_Results  --output_folder Flux2_Results_Plots
