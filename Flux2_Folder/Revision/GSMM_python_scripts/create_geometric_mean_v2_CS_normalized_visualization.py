#!/usr/bin/env python3
"""
Create Comprehensive Visualization for Geometric Mean v2 Method
Similar to calibrated_metabolism_visualization.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300

import os

print("="*80)
print("CREATING GEOMETRIC MEAN V2 CS-NORMALIZED VISUALIZATION WITH DATA EXPORT")
print("="*80)
print("\nLoading geometric mean v2 CS-normalized data...")

# Define paths
input_dir = 'geometric_mean_CS_normalized_output'
output_dir = 'geometric_mean_CS_normalized_output'

geom_df = pd.read_csv(os.path.join(input_dir, 'geometric_mean_v2_CS_normalized_pathway_fluxes.csv'))
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
main_pathways = ['Glycolysis', 'Oxidative_PPP', 'Non_oxidative_PPP',
                 'Lactate_Dehydrogenase', 'Pyruvate_Dehydrogenase', 'TCA_Cycle',
                 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis']

aa_pathways = ['Alanine_catabolism', 'Aspartate_catabolism', 'Leucine_catabolism',
               'Isoleucine_catabolism', 'Valine_catabolism', 'Serine_catabolism',
               'Glycine_catabolism', 'Threonine_catabolism', 'Methionine_catabolism',
               'Proline_catabolism', 'Arginine_catabolism', 'Histidine_catabolism',
               'Lysine_catabolism', 'Phenylalanine_catabolism', 'Tyrosine_catabolism',
               'Tryptophan_catabolism', 'Cysteine_catabolism', 'Asparagine_catabolism']

# Calculate tissue averages
tissue_avg = geom_df.groupby('Tissue')[main_pathways + aa_pathways].mean()

# Calculate color scale limits for CS-normalized data
# Use symmetric scale around 0 for diverging colormap
all_values = tissue_avg[main_pathways + aa_pathways].values.flatten()
vmax_main = max(abs(tissue_avg[main_pathways].values.flatten().min()),
                abs(tissue_avg[main_pathways].values.flatten().max()))
vmax_aa = max(abs(tissue_avg[aa_pathways].values.flatten().min()),
              abs(tissue_avg[aa_pathways].values.flatten().max()))

# Create figure
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)

# Color scheme
colors_main = ['#e74c3c', '#3498db', '#16a085', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
colors_aa = plt.cm.tab20(np.linspace(0, 1, len(aa_pathways)))

# ==================== PANEL 1: Main Pathways Heatmap ====================
ax1 = fig.add_subplot(gs[0, :])

# Sort tissues: non-brain alphabetically, then brain tissues alphabetically
brain_tissues = ['CB', 'Cor', 'HC', 'MB', 'ME', 'OB', 'Ps', 'Str', 'TH']
all_tissues = tissue_avg.index.tolist()
non_brain = sorted([t for t in all_tissues if t not in brain_tissues])
brain = sorted([t for t in all_tissues if t in brain_tissues])
tissue_order = non_brain + brain
heatmap_data = tissue_avg.loc[tissue_order, main_pathways].T

sns.heatmap(heatmap_data,
            cmap='RdBu_r',
            vmin=-vmax_main,
            vmax=vmax_main,
            center=0,
            cbar_kws={'label': 'CS-Normalized Pathway Activity', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='white',
            ax=ax1)

ax1.set_title('A. Main Metabolic Pathways by Tissue (CS-Normalized)',
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Tissue', fontsize=12)
ax1.set_ylabel('Pathway', fontsize=12)
ax1.set_yticklabels([p.replace('_', ' ') for p in main_pathways], rotation=0)

# Export Panel A data
heatmap_data.to_csv(os.path.join(output_dir, 'panel_A_main_pathways_heatmap.csv'))
print("✓ Exported: panel_A_main_pathways_heatmap.csv")

# Save Panel A as individual image
fig_a = plt.figure(figsize=(12, 6))
ax_a = fig_a.add_subplot(111)
sns.heatmap(heatmap_data,
            cmap='RdBu_r',
            vmin=-vmax_main,
            vmax=vmax_main,
            center=0,
            cbar_kws={'label': 'CS-Normalized Pathway Activity'},
            linewidths=0.5,
            linecolor='white',
            ax=ax_a)
ax_a.set_title('A. Main Metabolic Pathways by Tissue (CS-Normalized)',
              fontsize=14, fontweight='bold', pad=20)
ax_a.set_xlabel('Tissue', fontsize=12)
ax_a.set_ylabel('Pathway', fontsize=12)
ax_a.set_yticklabels([p.replace('_', ' ') for p in main_pathways], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_A_main_pathways_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig_a)
print("✓ Saved: panel_A_main_pathways_heatmap.png")

# ==================== PANEL 2: Amino Acid Pathways Heatmap ====================
ax2 = fig.add_subplot(gs[1, :])

heatmap_data_aa = tissue_avg.loc[tissue_order, aa_pathways].T

sns.heatmap(heatmap_data_aa,
            cmap='RdBu_r',
            vmin=-vmax_aa,
            vmax=vmax_aa,
            center=0,
            cbar_kws={'label': 'CS-Normalized Pathway Activity', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='white',
            yticklabels=[p.replace('_catabolism', '').replace('_', ' ') for p in aa_pathways],
            ax=ax2)

ax2.set_title('B. Amino Acid Catabolism Pathways by Tissue (CS-Normalized)',
              fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Tissue', fontsize=12)
ax2.set_ylabel('Amino Acid Pathway', fontsize=12)

# Export Panel B data
heatmap_data_aa.to_csv(os.path.join(output_dir, 'panel_B_amino_acid_pathways_heatmap.csv'))
print("✓ Exported: panel_B_amino_acid_pathways_heatmap.csv")

# Save Panel B as individual image
fig_b = plt.figure(figsize=(12, 10))
ax_b = fig_b.add_subplot(111)
sns.heatmap(heatmap_data_aa,
            cmap='RdBu_r',
            vmin=-vmax_aa,
            vmax=vmax_aa,
            center=0,
            cbar_kws={'label': 'CS-Normalized Pathway Activity'},
            linewidths=0.5,
            linecolor='white',
            yticklabels=[p.replace('_catabolism', '').replace('_', ' ') for p in aa_pathways],
            ax=ax_b)
ax_b.set_title('B. Amino Acid Catabolism Pathways by Tissue (CS-Normalized)',
              fontsize=14, fontweight='bold', pad=20)
ax_b.set_xlabel('Tissue', fontsize=12)
ax_b.set_ylabel('Amino Acid Pathway', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_B_amino_acid_pathways_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig_b)
print("✓ Saved: panel_B_amino_acid_pathways_heatmap.png")

# ==================== PANEL 3: Top Tissues by Pathway ====================
ax3 = fig.add_subplot(gs[2, 0])

top_tissues_data = []
for pathway in main_pathways:
    top_tissue = tissue_avg[pathway].idxmax()
    top_value = tissue_avg[pathway].max()
    top_tissues_data.append({'Pathway': pathway.replace('_', ' '),
                            'Tissue': top_tissue,
                            'Capacity': top_value})

top_df = pd.DataFrame(top_tissues_data)
y_pos = np.arange(len(top_df))

bars = ax3.barh(y_pos, top_df['Capacity'], color=colors_main, alpha=0.8, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_df['Pathway'])
ax3.set_xlabel('Pathway Capacity (0-1 scale)', fontsize=11)
ax3.set_title('C. Top Tissue for Each Main Pathway', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 1)
ax3.grid(axis='x', alpha=0.3)

# Add tissue labels
for i, (idx, row) in enumerate(top_df.iterrows()):
    ax3.text(row['Capacity'] + 0.02, i, f"{row['Tissue']}",
            va='center', fontsize=9, fontweight='bold')

# Export Panel C data
top_df.to_csv(os.path.join(output_dir, 'panel_C_top_tissues_by_pathway.csv'), index=False)
print("✓ Exported: panel_C_top_tissues_by_pathway.csv")

# Save Panel C as individual image
fig_c = plt.figure(figsize=(8, 6))
ax_c = fig_c.add_subplot(111)
y_pos_c = np.arange(len(top_df))
bars_c = ax_c.barh(y_pos_c, top_df['Capacity'], color=colors_main, alpha=0.8, edgecolor='black')
ax_c.set_yticks(y_pos_c)
ax_c.set_yticklabels(top_df['Pathway'])
ax_c.set_xlabel('Pathway Capacity (0-1 scale)', fontsize=11)
ax_c.set_title('C. Top Tissue for Each Main Pathway', fontsize=12, fontweight='bold')
ax_c.set_xlim(0, 1)
ax_c.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_df.iterrows()):
    ax_c.text(row['Capacity'] + 0.02, i, f"{row['Tissue']}",
            va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_C_top_tissues_by_pathway.png'), dpi=300, bbox_inches='tight')
plt.close(fig_c)
print("✓ Saved: panel_C_top_tissues_by_pathway.png")

# ==================== PANEL 4: Pathway Distribution ====================
ax4 = fig.add_subplot(gs[2, 1])

pathway_means = [geom_df[p].mean() for p in main_pathways]
pathway_stds = [geom_df[p].std() for p in main_pathways]

x_pos = np.arange(len(main_pathways))
bars = ax4.bar(x_pos, pathway_means, yerr=pathway_stds,
              color=colors_main, alpha=0.8, edgecolor='black',
              capsize=5, error_kw={'linewidth': 2})

ax4.set_xticks(x_pos)
ax4.set_xticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Mean Capacity (0-1 scale)', fontsize=11)
ax4.set_title('D. Mean Pathway Capacity Across All Samples',
             fontsize=12, fontweight='bold')
ax4.set_ylim(0, 0.6)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(pathway_means, pathway_stds)):
    ax4.text(i, mean + std + 0.02, f'{mean:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Export Panel D data
panel_d_df = pd.DataFrame({
    'Pathway': main_pathways,
    'Mean': pathway_means,
    'SD': pathway_stds
})
panel_d_df.to_csv(os.path.join(output_dir, 'panel_D_pathway_capacity_distribution.csv'), index=False)
print("✓ Exported: panel_D_pathway_capacity_distribution.csv")

# Save Panel D as individual image
fig_d = plt.figure(figsize=(10, 6))
ax_d = fig_d.add_subplot(111)
x_pos_d = np.arange(len(main_pathways))
bars_d = ax_d.bar(x_pos_d, pathway_means, yerr=pathway_stds,
              color=colors_main, alpha=0.8, edgecolor='black',
              capsize=5, error_kw={'linewidth': 2})
ax_d.set_xticks(x_pos_d)
ax_d.set_xticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=45, ha='right', fontsize=9)
ax_d.set_ylabel('Mean Capacity (0-1 scale)', fontsize=11)
ax_d.set_title('D. Mean Pathway Capacity Across All Samples',
             fontsize=12, fontweight='bold')
ax_d.set_ylim(0, 0.6)
ax_d.grid(axis='y', alpha=0.3)
for i, (mean, std) in enumerate(zip(pathway_means, pathway_stds)):
    ax_d.text(i, mean + std + 0.02, f'{mean:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_D_pathway_capacity_distribution.png'), dpi=300, bbox_inches='tight')
plt.close(fig_d)
print("✓ Saved: panel_D_pathway_capacity_distribution.png")

# ==================== PANEL 5: Tissue Clustering ====================
ax5 = fig.add_subplot(gs[3, :])

# Calculate pairwise correlations between tissues
tissue_corr = tissue_avg[main_pathways].T.corr()

# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

linkage_matrix = linkage(pdist(tissue_avg[main_pathways], metric='euclidean'),
                        method='ward')

# Plot dendrogram and heatmap
dendro = dendrogram(linkage_matrix, labels=tissue_avg.index, ax=ax5,
                   leaf_font_size=9, color_threshold=0)
ax5.set_title('E. Tissue Clustering by Metabolic Profile (Main Pathways)',
             fontsize=12, fontweight='bold')
ax5.set_xlabel('Tissue', fontsize=11)
ax5.set_ylabel('Distance', fontsize=11)

# Export Panel E data
panel_e_linkage = pd.DataFrame(linkage_matrix,
                                columns=['Cluster1', 'Cluster2', 'Distance', 'NumSamples'])
panel_e_linkage.to_csv(os.path.join(output_dir, 'panel_E_tissue_clustering_linkage.csv'), index=False)
tissue_labels_df = pd.DataFrame({
    'Index': range(len(tissue_avg.index)),
    'Tissue': tissue_avg.index
})
tissue_labels_df.to_csv(os.path.join(output_dir, 'panel_E_tissue_clustering_labels.csv'), index=False)
tissue_avg[main_pathways].to_csv(os.path.join(output_dir, 'panel_E_tissue_metabolic_profiles.csv'))
print("✓ Exported: panel_E_tissue_clustering_linkage.csv")
print("✓ Exported: panel_E_tissue_clustering_labels.csv")
print("✓ Exported: panel_E_tissue_metabolic_profiles.csv")

# Save Panel E as individual image
fig_e = plt.figure(figsize=(12, 6))
ax_e = fig_e.add_subplot(111)
dendro_e = dendrogram(linkage_matrix, labels=tissue_avg.index, ax=ax_e,
                   leaf_font_size=9, color_threshold=0)
ax_e.set_title('E. Tissue Clustering by Metabolic Profile (Main Pathways)',
             fontsize=12, fontweight='bold')
ax_e.set_xlabel('Tissue', fontsize=11)
ax_e.set_ylabel('Distance', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_E_tissue_clustering.png'), dpi=300, bbox_inches='tight')
plt.close(fig_e)
print("✓ Saved: panel_E_tissue_clustering.png")

# ==================== PANEL 6: Pathway Correlations ====================
ax6 = fig.add_subplot(gs[4, 0])

# Calculate pathway correlations
pathway_corr = geom_df[main_pathways].corr()

sns.heatmap(pathway_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            ax=ax6)

ax6.set_title('F. Pathway Co-regulation Patterns',
             fontsize=12, fontweight='bold')
ax6.set_xticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=45, ha='right', fontsize=8)
ax6.set_yticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=0, fontsize=8)

# Export Panel F data
pathway_corr.to_csv(os.path.join(output_dir, 'panel_F_pathway_correlations.csv'))
print("✓ Exported: panel_F_pathway_correlations.csv")

# Save Panel F as individual image
fig_f = plt.figure(figsize=(8, 7))
ax_f = fig_f.add_subplot(111)
sns.heatmap(pathway_corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation'},
            ax=ax_f)
ax_f.set_title('F. Pathway Co-regulation Patterns',
             fontsize=12, fontweight='bold')
ax_f.set_xticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=45, ha='right', fontsize=9)
ax_f.set_yticklabels([p.replace('_', ' ') for p in main_pathways],
                    rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_F_pathway_correlations.png'), dpi=300, bbox_inches='tight')
plt.close(fig_f)
print("✓ Saved: panel_F_pathway_correlations.png")

# ==================== PANEL 7: Sex Differences ====================
ax7 = fig.add_subplot(gs[4, 1])

# Calculate sex differences for tissues with both sexes
sex_diff_data = []
for tissue in geom_df['Tissue'].dropna().unique():
    tissue_data = geom_df[geom_df['Tissue'] == tissue]
    male_data = tissue_data[tissue_data['Sex'] == 'Male']
    female_data = tissue_data[tissue_data['Sex'] == 'Female']

    if len(male_data) > 0 and len(female_data) > 0:
        for pathway in main_pathways:
            diff = male_data[pathway].mean() - female_data[pathway].mean()
            sex_diff_data.append({
                'Tissue': tissue,
                'Pathway': pathway.replace('_', ' '),
                'Difference': diff
            })

sex_diff_df = pd.DataFrame(sex_diff_data)

# Plot top 10 largest sex differences
sex_diff_df['AbsDiff'] = sex_diff_df['Difference'].abs()
top_sex_diff = sex_diff_df.nlargest(10, 'AbsDiff')

colors_sex = ['crimson' if x > 0 else 'steelblue' for x in top_sex_diff['Difference']]
y_pos = np.arange(len(top_sex_diff))

ax7.barh(y_pos, top_sex_diff['Difference'], color=colors_sex, alpha=0.7, edgecolor='black')
ax7.set_yticks(y_pos)
ax7.set_yticklabels([f"{row['Tissue']} - {row['Pathway']}"
                     for _, row in top_sex_diff.iterrows()], fontsize=8)
ax7.axvline(0, color='black', linewidth=2)
ax7.set_xlabel('Sex Difference (Male - Female)', fontsize=11)
ax7.set_title('G. Top 10 Sex Differences in Pathway Capacity',
             fontsize=12, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='crimson', alpha=0.7, label='Male > Female'),
                  Patch(facecolor='steelblue', alpha=0.7, label='Female > Male')]
ax7.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Export Panel G data
sex_diff_df.to_csv(os.path.join(output_dir, 'panel_G_sex_differences_all.csv'), index=False)
top_sex_diff.to_csv(os.path.join(output_dir, 'panel_G_sex_differences_top10.csv'), index=False)
print("✓ Exported: panel_G_sex_differences_all.csv")
print("✓ Exported: panel_G_sex_differences_top10.csv")

# Save Panel G as individual image
fig_g = plt.figure(figsize=(10, 6))
ax_g = fig_g.add_subplot(111)
colors_sex_g = ['crimson' if x > 0 else 'steelblue' for x in top_sex_diff['Difference']]
y_pos_g = np.arange(len(top_sex_diff))
ax_g.barh(y_pos_g, top_sex_diff['Difference'], color=colors_sex_g, alpha=0.7, edgecolor='black')
ax_g.set_yticks(y_pos_g)
ax_g.set_yticklabels([f"{row['Tissue']} - {row['Pathway']}"
                     for _, row in top_sex_diff.iterrows()], fontsize=9)
ax_g.axvline(0, color='black', linewidth=2)
ax_g.set_xlabel('Sex Difference (Male - Female)', fontsize=11)
ax_g.set_title('G. Top 10 Sex Differences in Pathway Capacity',
             fontsize=12, fontweight='bold')
ax_g.grid(axis='x', alpha=0.3)
legend_elements_g = [Patch(facecolor='crimson', alpha=0.7, label='Male > Female'),
                    Patch(facecolor='steelblue', alpha=0.7, label='Female > Male')]
ax_g.legend(handles=legend_elements_g, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_G_sex_differences.png'), dpi=300, bbox_inches='tight')
plt.close(fig_g)
print("✓ Saved: panel_G_sex_differences.png")

# ==================== PANEL 8: Summary Statistics ====================
ax8 = fig.add_subplot(gs[5, :])
ax8.axis('off')

# Create summary table
summary_stats = []
for pathway in main_pathways:
    values = geom_df[pathway]
    summary_stats.append([
        pathway.replace('_', ' '),
        f'{values.mean():.3f}',
        f'{values.median():.3f}',
        f'{values.std():.3f}',
        f'{values.min():.3f}',
        f'{values.max():.3f}',
        f'{len(values)}'
    ])

table = ax8.table(cellText=summary_stats,
                 colLabels=['Pathway', 'Mean', 'Median', 'SD', 'Min', 'Max', 'N'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_stats) + 1):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('white')

ax8.set_title('H. Summary Statistics for Main Pathways',
             fontsize=12, fontweight='bold', pad=20)

# Export Panel H data
panel_h_df = pd.DataFrame(summary_stats,
                          columns=['Pathway', 'Mean', 'Median', 'SD', 'Min', 'Max', 'N'])
panel_h_df.to_csv(os.path.join(output_dir, 'panel_H_summary_statistics.csv'), index=False)
print("✓ Exported: panel_H_summary_statistics.csv")

# Save Panel H as individual image
fig_h = plt.figure(figsize=(12, 4))
ax_h = fig_h.add_subplot(111)
ax_h.axis('off')
table_h = ax_h.table(cellText=summary_stats,
                 colLabels=['Pathway', 'Mean', 'Median', 'SD', 'Min', 'Max', 'N'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table_h.auto_set_font_size(False)
table_h.set_fontsize(10)
table_h.scale(1, 2.5)
# Style header
for i in range(7):
    table_h[(0, i)].set_facecolor('#3498db')
    table_h[(0, i)].set_text_props(weight='bold', color='white')
# Alternate row colors
for i in range(1, len(summary_stats) + 1):
    for j in range(7):
        if i % 2 == 0:
            table_h[(i, j)].set_facecolor('#ecf0f1')
        else:
            table_h[(i, j)].set_facecolor('white')
ax_h.set_title('H. Summary Statistics for Main Pathways',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'panel_H_summary_statistics.png'), dpi=300, bbox_inches='tight')
plt.close(fig_h)
print("✓ Saved: panel_H_summary_statistics.png")

# ==================== Overall Title ====================
fig.suptitle('Geometric Mean v2 CS-Normalized: Comprehensive Metabolic Pathway Analysis\n' +
            f'245 Mouse Tissue Samples | 29 Pathways | CS-Normalized (TCA without CS)',
            fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.savefig(os.path.join(output_dir, 'geometric_mean_v2_CS_normalized_visualization.png'), dpi=300, bbox_inches='tight')
print("\n✓ Saved comprehensive visualization to geometric_mean_v2_CS_normalized_visualization.png")
plt.close()

# ==================== BONUS: Export Additional Useful Data ====================
print("\n" + "="*80)
print("EXPORTING BONUS DATA FILES")
print("="*80)

# Individual sample data with metadata
sample_data = geom_df[['Sample', 'Name', 'Tissue', 'Sex'] + main_pathways + aa_pathways]
sample_data.to_csv(os.path.join(output_dir, 'all_samples_with_metadata.csv'), index=False)
print("✓ Exported: all_samples_with_metadata.csv")

# Tissue averages (all pathways)
tissue_avg_all = geom_df.groupby('Tissue')[main_pathways + aa_pathways].mean()
tissue_avg_all.to_csv(os.path.join(output_dir, 'tissue_averages_all_pathways.csv'))
print("✓ Exported: tissue_averages_all_pathways.csv")

# Tissue-sex averages
tissue_sex_avg = geom_df.groupby(['Tissue', 'Sex'])[main_pathways + aa_pathways].mean()
tissue_sex_avg.to_csv(os.path.join(output_dir, 'tissue_sex_averages_all_pathways.csv'))
print("✓ Exported: tissue_sex_averages_all_pathways.csv")

print("\n" + "="*80)
print("VISUALIZATION AND DATA EXPORT COMPLETE")
print("="*80)
print("\nGenerated visualizations:")
print("  • geometric_mean_v2_visualization.png (combined 8-panel figure)")
print("\nIndividual panel images:")
print("  • panel_A_main_pathways_heatmap.png")
print("  • panel_B_amino_acid_pathways_heatmap.png")
print("  • panel_C_top_tissues_by_pathway.png")
print("  • panel_D_pathway_capacity_distribution.png")
print("  • panel_E_tissue_clustering.png")
print("  • panel_F_pathway_correlations.png")
print("  • panel_G_sex_differences.png")
print("  • panel_H_summary_statistics.png")
print("\nExported panel data files (CSV):")
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
print("\nBonus data files:")
print("  • all_samples_with_metadata.csv (245 samples)")
print("  • tissue_averages_all_pathways.csv (tissue means)")
print("  • tissue_sex_averages_all_pathways.csv (tissue-sex means)")
print("\n" + "="*80)
