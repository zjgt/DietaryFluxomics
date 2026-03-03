#!/usr/bin/env python3
"""
Export Panel-by-Panel Data from Geometric Mean v2 Visualization
Generates separate CSV files for each panel for easier editing
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

print("="*80)
print("EXPORTING GEOMETRIC MEAN V2 VISUALIZATION PANEL DATA")
print("="*80)

# Load data
print("\nLoading geometric mean v2 data...")
geom_df = pd.read_csv('geometric_mean_v2_pathway_fluxes.csv')
sample_info = pd.read_csv('Sample_Info.csv')

# Merge with sample info
geom_df = geom_df.merge(sample_info[['Sample', 'Name', 'Description']],
                        left_on='Sample', right_on='Sample', how='left')

# Parse tissue and sex
def parse_name(name):
    if pd.isna(name):
        return None, None
    parts = name.split('_')
    tissue = parts[0] if len(parts) > 0 else None
    sex = parts[2] if len(parts) > 2 else None
    return tissue, sex

geom_df[['Tissue', 'Sex']] = geom_df['Name'].apply(lambda x: pd.Series(parse_name(x)))
geom_df['Sex'] = geom_df['Sex'].map({'F': 'Female', 'M': 'Male'})

# Define pathway columns
main_pathways = ['Glycolysis', 'Oxidative_PPP', 'TCA_Cycle', 'Pyruvate_Dehydrogenase',
                 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis']

aa_pathways = ['Alanine_catabolism', 'Aspartate_catabolism', 'Leucine_catabolism',
               'Isoleucine_catabolism', 'Valine_catabolism', 'Serine_catabolism',
               'Glycine_catabolism', 'Threonine_catabolism', 'Methionine_catabolism',
               'Proline_catabolism', 'Arginine_catabolism', 'Histidine_catabolism',
               'Lysine_catabolism', 'Phenylalanine_catabolism', 'Tyrosine_catabolism',
               'Tryptophan_catabolism', 'Cysteine_catabolism', 'Asparagine_catabolism']

# Calculate tissue averages
tissue_avg = geom_df.groupby('Tissue')[main_pathways + aa_pathways].mean()

# Sort tissues by overall metabolic activity
tissue_order = tissue_avg[main_pathways].mean(axis=1).sort_values(ascending=False).index

# ==================== PANEL A: Main Pathways Heatmap ====================
print("\n" + "="*80)
print("PANEL A: Main Metabolic Pathways Heatmap")
print("="*80)

panel_a_data = tissue_avg.loc[tissue_order, main_pathways].T
panel_a_data.to_csv('panel_A_main_pathways_heatmap.csv')
print("✓ Saved: panel_A_main_pathways_heatmap.csv")
print(f"  Shape: {panel_a_data.shape} (rows=pathways, cols=tissues)")
print(f"  Tissues ordered by metabolic activity")

# ==================== PANEL B: Amino Acid Pathways Heatmap ====================
print("\n" + "="*80)
print("PANEL B: Amino Acid Catabolism Heatmap")
print("="*80)

panel_b_data = tissue_avg.loc[tissue_order, aa_pathways].T
panel_b_data.to_csv('panel_B_amino_acid_pathways_heatmap.csv')
print("✓ Saved: panel_B_amino_acid_pathways_heatmap.csv")
print(f"  Shape: {panel_b_data.shape} (rows=pathways, cols=tissues)")

# ==================== PANEL C: Top Tissues by Pathway ====================
print("\n" + "="*80)
print("PANEL C: Top Tissue for Each Main Pathway")
print("="*80)

panel_c_data = []
for pathway in main_pathways:
    top_tissue = tissue_avg[pathway].idxmax()
    top_value = tissue_avg[pathway].max()
    panel_c_data.append({
        'Pathway': pathway,
        'Top_Tissue': top_tissue,
        'Capacity': top_value
    })

panel_c_df = pd.DataFrame(panel_c_data)
panel_c_df.to_csv('panel_C_top_tissues_by_pathway.csv', index=False)
print("✓ Saved: panel_C_top_tissues_by_pathway.csv")
print(f"  Shape: {panel_c_df.shape}")

# ==================== PANEL D: Pathway Distribution ====================
print("\n" + "="*80)
print("PANEL D: Mean Pathway Capacity Distribution")
print("="*80)

panel_d_data = []
for pathway in main_pathways:
    values = geom_df[pathway]
    panel_d_data.append({
        'Pathway': pathway,
        'Mean': values.mean(),
        'SD': values.std(),
        'Median': values.median(),
        'Min': values.min(),
        'Max': values.max(),
        'N': len(values)
    })

panel_d_df = pd.DataFrame(panel_d_data)
panel_d_df.to_csv('panel_D_pathway_capacity_distribution.csv', index=False)
print("✓ Saved: panel_D_pathway_capacity_distribution.csv")
print(f"  Shape: {panel_d_df.shape}")

# ==================== PANEL E: Tissue Clustering ====================
print("\n" + "="*80)
print("PANEL E: Tissue Clustering by Metabolic Profile")
print("="*80)

# Distance matrix
dist_matrix = pdist(tissue_avg[main_pathways], metric='euclidean')
linkage_matrix = linkage(dist_matrix, method='ward')

# Create dendrogram data
panel_e_linkage = pd.DataFrame(linkage_matrix,
                                columns=['Cluster1', 'Cluster2', 'Distance', 'NumSamples'])
panel_e_linkage['Cluster1'] = panel_e_linkage['Cluster1'].astype(int)
panel_e_linkage['Cluster2'] = panel_e_linkage['Cluster2'].astype(int)
panel_e_linkage['NumSamples'] = panel_e_linkage['NumSamples'].astype(int)

# Add tissue labels
tissue_labels = pd.DataFrame({
    'Index': range(len(tissue_avg.index)),
    'Tissue': tissue_avg.index
})

panel_e_linkage.to_csv('panel_E_tissue_clustering_linkage.csv', index=False)
tissue_labels.to_csv('panel_E_tissue_clustering_labels.csv', index=False)
print("✓ Saved: panel_E_tissue_clustering_linkage.csv")
print("✓ Saved: panel_E_tissue_clustering_labels.csv")
print(f"  Linkage shape: {panel_e_linkage.shape}")

# Also save the distance matrix for reference
tissue_avg[main_pathways].to_csv('panel_E_tissue_metabolic_profiles.csv')
print("✓ Saved: panel_E_tissue_metabolic_profiles.csv (raw data for clustering)")

# ==================== PANEL F: Pathway Correlations ====================
print("\n" + "="*80)
print("PANEL F: Pathway Co-regulation Patterns")
print("="*80)

pathway_corr = geom_df[main_pathways].corr()
pathway_corr.to_csv('panel_F_pathway_correlations.csv')
print("✓ Saved: panel_F_pathway_correlations.csv")
print(f"  Shape: {pathway_corr.shape}")

# ==================== PANEL G: Sex Differences ====================
print("\n" + "="*80)
print("PANEL G: Top 10 Sex Differences in Pathway Capacity")
print("="*80)

sex_diff_data = []
for tissue in geom_df['Tissue'].dropna().unique():
    tissue_data = geom_df[geom_df['Tissue'] == tissue]
    male_data = tissue_data[tissue_data['Sex'] == 'Male']
    female_data = tissue_data[tissue_data['Sex'] == 'Female']

    if len(male_data) > 0 and len(female_data) > 0:
        for pathway in main_pathways:
            male_mean = male_data[pathway].mean()
            female_mean = female_data[pathway].mean()
            diff = male_mean - female_mean
            pct_diff = (diff / female_mean * 100) if female_mean != 0 else 0

            sex_diff_data.append({
                'Tissue': tissue,
                'Pathway': pathway,
                'Male_Mean': male_mean,
                'Female_Mean': female_mean,
                'Difference': diff,
                'Percent_Difference': pct_diff,
                'Male_N': len(male_data),
                'Female_N': len(female_data)
            })

sex_diff_df = pd.DataFrame(sex_diff_data)
sex_diff_df['AbsDiff'] = sex_diff_df['Difference'].abs()

# Full data
sex_diff_df.to_csv('panel_G_sex_differences_all.csv', index=False)
print("✓ Saved: panel_G_sex_differences_all.csv (all tissue-pathway combinations)")
print(f"  Shape: {sex_diff_df.shape}")

# Top 10 for plotting
top_sex_diff = sex_diff_df.nlargest(10, 'AbsDiff')
top_sex_diff.to_csv('panel_G_sex_differences_top10.csv', index=False)
print("✓ Saved: panel_G_sex_differences_top10.csv (top 10 for plot)")
print(f"  Shape: {top_sex_diff.shape}")

# ==================== PANEL H: Summary Statistics ====================
print("\n" + "="*80)
print("PANEL H: Summary Statistics for Main Pathways")
print("="*80)

summary_stats = []
for pathway in main_pathways:
    values = geom_df[pathway]
    summary_stats.append({
        'Pathway': pathway,
        'Mean': values.mean(),
        'Median': values.median(),
        'SD': values.std(),
        'Min': values.min(),
        'Max': values.max(),
        'N': len(values)
    })

panel_h_df = pd.DataFrame(summary_stats)
panel_h_df.to_csv('panel_H_summary_statistics.csv', index=False)
print("✓ Saved: panel_H_summary_statistics.csv")
print(f"  Shape: {panel_h_df.shape}")

# ==================== BONUS: Individual Sample Data ====================
print("\n" + "="*80)
print("BONUS: Individual Sample Data (for custom plots)")
print("="*80)

# Export sample-level data with metadata
sample_data = geom_df[['Sample', 'Name', 'Tissue', 'Sex'] + main_pathways + aa_pathways]
sample_data.to_csv('all_samples_with_metadata.csv', index=False)
print("✓ Saved: all_samples_with_metadata.csv")
print(f"  Shape: {sample_data.shape}")
print(f"  Contains: All 245 samples with tissue/sex metadata and pathway values")

# Export tissue averages (all pathways)
tissue_avg_all = geom_df.groupby('Tissue')[main_pathways + aa_pathways].mean()
tissue_avg_all.to_csv('tissue_averages_all_pathways.csv')
print("✓ Saved: tissue_averages_all_pathways.csv")
print(f"  Shape: {tissue_avg_all.shape}")

# Export tissue-sex averages
tissue_sex_avg = geom_df.groupby(['Tissue', 'Sex'])[main_pathways + aa_pathways].mean()
tissue_sex_avg.to_csv('tissue_sex_averages_all_pathways.csv')
print("✓ Saved: tissue_sex_averages_all_pathways.csv")
print(f"  Shape: {tissue_sex_avg.shape}")

print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print("\nGenerated files for each panel:")
print("  Panel A: panel_A_main_pathways_heatmap.csv")
print("  Panel B: panel_B_amino_acid_pathways_heatmap.csv")
print("  Panel C: panel_C_top_tissues_by_pathway.csv")
print("  Panel D: panel_D_pathway_capacity_distribution.csv")
print("  Panel E: panel_E_tissue_clustering_linkage.csv")
print("           panel_E_tissue_clustering_labels.csv")
print("           panel_E_tissue_metabolic_profiles.csv")
print("  Panel F: panel_F_pathway_correlations.csv")
print("  Panel G: panel_G_sex_differences_all.csv")
print("           panel_G_sex_differences_top10.csv")
print("  Panel H: panel_H_summary_statistics.csv")
print("\nBonus files:")
print("  - all_samples_with_metadata.csv (individual samples)")
print("  - tissue_averages_all_pathways.csv (tissue means)")
print("  - tissue_sex_averages_all_pathways.csv (tissue-sex means)")
print("\n" + "="*80)
