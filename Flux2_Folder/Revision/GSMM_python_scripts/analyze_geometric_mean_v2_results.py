#!/usr/bin/env python3
"""
Analyze Geometric Mean v2 Results
- Calculate tissue-sex averages
- Calculate sex differences
- Generate visualizations
- Create summary reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
print("Loading geometric mean v2 results...")
geom_df = pd.read_csv('geometric_mean_v2_pathway_fluxes.csv')
sample_info = pd.read_csv('Sample_Info.csv')

# Parse tissue and sex from sample names
print("Parsing sample metadata...")
geom_df = geom_df.merge(sample_info[['Sample', 'Name', 'Description']], on='Sample', how='left')

# Extract tissue and sex from Name column
def parse_name(name):
    if pd.isna(name):
        return None, None
    parts = name.split('_')
    tissue = parts[0] if len(parts) > 0 else None
    sex = parts[2] if len(parts) > 2 else None
    return tissue, sex

geom_df[['Tissue', 'Sex']] = geom_df['Name'].apply(lambda x: pd.Series(parse_name(x)))

# Map sex codes
geom_df['Sex'] = geom_df['Sex'].map({'F': 'Female', 'M': 'Male'})

print(f"Found {geom_df['Tissue'].nunique()} tissues")
print(f"Samples per sex: {geom_df['Sex'].value_counts().to_dict()}")

# Calculate tissue-sex averages
print("\nCalculating tissue-sex averages...")
pathway_cols = [col for col in geom_df.columns if col not in ['Sample', 'Name', 'Description', 'Tissue', 'Sex']]

tissue_sex_avg = geom_df.groupby(['Tissue', 'Sex'])[pathway_cols].agg(['mean', 'sem', 'count']).reset_index()

# Flatten column names
tissue_sex_avg.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                          for col in tissue_sex_avg.columns.values]

# Rename for clarity
tissue_sex_avg = tissue_sex_avg.rename(columns={
    'Tissue': 'tissue',
    'Sex': 'sex'
})

# Save tissue-sex averages
tissue_sex_avg.to_csv('geometric_mean_v2_tissue_sex_averages.csv', index=False)
print(f"✓ Saved tissue-sex averages: {tissue_sex_avg.shape}")

# Calculate sex differences
print("\nCalculating sex differences...")
sex_diff_results = []

for tissue in geom_df['Tissue'].dropna().unique():
    tissue_data = geom_df[geom_df['Tissue'] == tissue]

    male_data = tissue_data[tissue_data['Sex'] == 'Male']
    female_data = tissue_data[tissue_data['Sex'] == 'Female']

    if len(male_data) > 0 and len(female_data) > 0:
        result = {'Tissue': tissue,
                  'Male_n': len(male_data),
                  'Female_n': len(female_data)}

        for col in pathway_cols:
            male_mean = male_data[col].mean()
            female_mean = female_data[col].mean()
            diff = male_mean - female_mean

            result[f'{col}_Male'] = male_mean
            result[f'{col}_Female'] = female_mean
            result[f'{col}_Diff'] = diff

        sex_diff_results.append(result)

sex_diff_df = pd.DataFrame(sex_diff_results)
sex_diff_df.to_csv('geometric_mean_v2_sex_differences.csv', index=False)
print(f"✓ Saved sex differences: {sex_diff_df.shape}")

# Generate summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary_stats = []
for col in ['Glycolysis', 'Oxidative_PPP', 'TCA_Cycle', 'Pyruvate_Dehydrogenase',
            'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis']:
    values = geom_df[col].dropna()
    summary_stats.append({
        'Pathway': col,
        'Mean': values.mean(),
        'Median': values.median(),
        'SD': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'N': len(values)
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))
summary_df.to_csv('geometric_mean_v2_summary_statistics.csv', index=False)

# Top tissues for each pathway
print("\n" + "="*80)
print("TOP 5 TISSUES BY PATHWAY (Geometric Mean v2)")
print("="*80)

for pathway in ['Glycolysis', 'TCA_Cycle', 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis']:
    tissue_means = geom_df.groupby('Tissue')[pathway].mean().sort_values(ascending=False)
    print(f"\n{pathway}:")
    for i, (tissue, value) in enumerate(tissue_means.head(5).items(), 1):
        print(f"  {i}. {tissue}: {value:.3f}")

print("\n✓ Analysis complete!")
print(f"\nOutput files:")
print(f"  - geometric_mean_v2_tissue_sex_averages.csv")
print(f"  - geometric_mean_v2_sex_differences.csv")
print(f"  - geometric_mean_v2_summary_statistics.csv")
