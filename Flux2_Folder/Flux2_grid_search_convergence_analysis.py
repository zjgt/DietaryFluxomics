import pandas as pd
from scipy import stats
import os
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to calculate confidence interval
def calc_conf_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


# Function to process each CSV file
def process_csv_file(file_path, output_folder):
    try:
        #df = pd.read_csv(file_path, header=0, index_col=0) #make sure the column names do not contain '-'.
        df = pd.read_csv(file_path, header=0)
        #df.fillna('NA', inplace=True)
        df.set_index(df.columns[0], inplace=True)
        df = df.drop(columns=['Reaction', 'External ids'])
        df = df.drop(index=['Thres', 'p_value', 'Id', 'r001_SubsGlc'])
        # delete second to fourth rows
        # use the first column as row indexes
        # delete all rows from the row named "Experiment" and onward
        if 'Experiment' in df.index:
            idx = df.index.get_loc('Experiment')
            df = df.iloc[:idx]
        df = df.T
        df = df.astype(float, errors='ignore')
        # Ensure 'RSS' is in the dataframe and not empty
        if 'RSS' not in df.columns or df['RSS'].empty:
            logging.warning(f"Skipping {file_path}: 'RSS' column missing or empty.")
            return

        # Calculate the threshold for the bottom 80% of RSS values
        rss_threshold = df['RSS'].quantile(0.8)

        # Filter the dataframe to include only rows with RSS values in the bottom 80%
        df_filtered = df[df['RSS'] <= rss_threshold]

        if df_filtered.empty:
            logging.warning(f"No rows with RSS values in the bottom 80% found in {file_path}.")
            return

        # Sort by RSS to find the smallest values
        df_sorted = df_filtered.sort_values(by='RSS')

        # Get the ten rows with the lowest RSS values
        lowest_rss_rows = df_sorted.head(10)

        if lowest_rss_rows.empty:
            logging.warning(f"No rows with low RSS values found in {file_path}.")
            return

        # Calculate projected values for each feature at the lowest RSS
        features = [col for col in df.columns if col != 'RSS']
        results = []
        for feature in features:
            # Calculate the mean of the ten lowest RSS values for the feature
            projected_value = lowest_rss_rows[feature].mean()

            # Calculate the confidence interval for the feature
            conf_mean, conf_lower, conf_upper = calc_conf_interval(lowest_rss_rows[feature])

            # Split data into bins based on RSS for convergence analysis
            num_bins = 5  # Can be adjusted or passed as a parameter
            df_sorted['bin'] = pd.cut(df_sorted['RSS'], bins=num_bins)
            # Calculate variance within each bin
            variances = df_sorted.groupby('bin')[feature].var()
            # Perform linear regression of variance vs. mean RSS within each bin
            bin_means = df_sorted.groupby('bin')['RSS'].mean().values
            variances_values = variances.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(bin_means, np.log(
                variances_values))  # Use log-variance for better linearity
            # Convergence test: Check if the slope of variance vs. RSS is positive and statistically significant
            convergence_test_result = "Converges" if slope > 0 and p_value < 0.05 else "Does not converge"

            # Append results
            results.append({
                'Feature': feature,
                'Projected Converge Value': projected_value,
                'Confidence Interval Lower': conf_lower,
                'Confidence Interval Upper': conf_upper,
                'Slope (Variance vs. RSS)': slope,
                'P-value': p_value,
                'R-squared': r_value ** 2,  # Coefficient of determination
                'Standard Error': std_err,
                'Convergence Test Result': convergence_test_result
            })

        if results:
            output_df = pd.DataFrame(results)
            file_name = os.path.basename(file_path).split('.')[0]
            output_file_path = os.path.join(output_folder, f"{file_name}_convergence_results.csv")
            # Save to specified output folder
            output_df.to_csv(output_file_path, index=False)
            logging.info(f"Results for {file_path} saved to {output_file_path}")
        else:
            logging.warning(f"No valid 'RSS' values found in {file_path}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")


# Main function with input and output folder parameters
def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                process_csv_file(file_path, output_folder)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform convergence tests on CSV files')
    # Add arguments for input folder and output folder
    parser.add_argument('--input', required=True, help='Input folder containing CSV files')
    parser.add_argument('--output', required=True, help='Output folder for results')
    args = parser.parse_args()
    main(args.input, args.output)



#python Flux2_grid_search_convergence_analysis.py --input Flux2_Results_Summary --output Flux2_Results_Convergence