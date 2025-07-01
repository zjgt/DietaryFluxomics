import pandas as pd
import glob
import os

def stitch_template(template_filename, folder_name):
    """
    Stitch the template csv file to the left side of each csv file in the
folder.

    Parameters:
        template_filename (str): The filename of the template csv file.
        folder_name (str): The name of the folder containing the csv files
to be stitched.

    Returns:
        None
    """

    # Load the template csv file into a pandas DataFrame
    template_df = pd.read_csv(template_filename)

    # Get a list of all csv files in the specified folder
    csv_files = glob.glob(os.path.join(folder_name, '*.csv'))

    # Iterate over each csv file in the folder
    for csv_file in csv_files:
        # Load the csv file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Stitch the template to the left side of the csv file
        stitched_df = pd.concat([template_df, df], axis=1)

        # Save the stitched DataFrame back to a new csv file
        output_filename = f"{os.path.basename(csv_file).split('.')[0]}_stitched.csv"
        stitched_df.to_csv(os.path.join(folder_name, output_filename), index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stitch template csv file to left side of each csv file in folder')
    parser.add_argument('template_filename', help='Filename of the template csv file')
    parser.add_argument('folder_name', help='Name of the folder containing the csv files to be stitched')

    args = parser.parse_args()

    stitch_template(args.template_filename, args.folder_name)


#python Flux2_template_summary_stitching.py Flux2_results_template_mouse.csv Flux2_Results_Summary
