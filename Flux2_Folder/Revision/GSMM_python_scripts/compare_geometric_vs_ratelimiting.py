#!/usr/bin/env python3
"""
Compare Geometric Mean v2 vs Rate-Limiting Method
- Calculate correlations for each pathway
- Generate scatter plots
- Create comparison tables
- Statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*80)
print("GEOMETRIC MEAN V2 vs RATE-LIMITING METHOD COMPARISON")
print("="*80)

# Load data
print("\nLoading data...")
geom_df = pd.read_csv('geometric_mean_v2_pathway_fluxes.csv')
rate_df = pd.read_csv('all_samples_calibrated_metabolism.csv')

print(f"Geometric Mean v2: {geom_df.shape}")
print(f"Rate-Limiting: {rate_df.shape}")

# Merge datasets on sample ID
# Geometric uses 'Sample', rate-limiting uses 'sample'
geom_df = geom_df.rename(columns={'Sample': 'sample'})
merged_df = geom_df.merge(rate_df, on='sample', how='inner')
print(f"Merged dataset: {merged_df.shape} ({len(merged_df)} samples)")

# Define pathway mappings (geometric v2 -> rate-limiting)
pathway_pairs = {
    'Glycolysis': 'Glycolysis_capacity',
    'Oxidative_PPP': 'Oxidative_PPP_capacity',
    'TCA_Cycle': 'TCA_Cycle_capacity',
    'Pyruvate_Dehydrogenase': 'Pyruvate_Dehydrogenase_capacity',
    'Fatty_Acid_Oxidation': 'Fatty_Acid_Oxidation_capacity',
    'Fatty_Acid_Synthesis': 'Fatty_Acid_Synthesis_capacity',
    'Glutaminolysis': 'Glutaminolysis_capacity',
    'Malic_Enzyme': 'Malic_Enzyme_capacity',
    'IDH_NADPH': 'NADPH_Isocitrate_DH_capacity',
    'Alanine_catabolism': 'Alanine_catabolism_capacity_cal',
    'Aspartate_catabolism': 'Aspartate_catabolism_capacity_cal',
    'Leucine_catabolism': 'Leucine_catabolism_capacity_cal',
    'Isoleucine_catabolism': 'Isoleucine_catabolism_capacity_cal',
    'Valine_catabolism': 'Valine_catabolism_capacity_cal'
}

# Calculate correlations
print("\n" + "="*80)
print("PATHWAY CORRELATIONS (Geometric Mean v2 vs Rate-Limiting)")
print("="*80)

correlation_results = []

for geom_col, rate_col in pathway_pairs.items():
    if geom_col in merged_df.columns and rate_col in merged_df.columns:
        # Remove any NaN or inf values
        valid_mask = (
            np.isfinite(merged_df[geom_col]) &
            np.isfinite(merged_df[rate_col]) &
            (merged_df[geom_col] > 0) &
            (merged_df[rate_col] > 0)
        )

        x = merged_df.loc[valid_mask, geom_col]
        y = merged_df.loc[valid_mask, rate_col]

        if len(x) > 3:
            r, p = pearsonr(x, y)

            # Calculate means for scale comparison
            geom_mean = x.mean()
            rate_mean = y.mean()
            scale_ratio = rate_mean / geom_mean if geom_mean > 0 else np.nan

            correlation_results.append({
                'Pathway': geom_col,
                'Pearson_r': r,
                'P_value': p,
                'N_samples': len(x),
                'Geom_Mean': geom_mean,
                'Rate_Mean': rate_mean,
                'Scale_Ratio': scale_ratio
            })

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{geom_col:30s}: r = {r:6.3f} ({sig}), n = {len(x):3d}, scale ratio = {scale_ratio:6.1f}x")

corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values('Pearson_r', ascending=False)
corr_df.to_csv('method_comparison_correlations.csv', index=False)
print(f"\n✓ Saved correlations to method_comparison_correlations.csv")

# Summary statistics
print("\n" + "="*80)
print("CORRELATION SUMMARY")
print("="*80)
print(f"Mean correlation: {corr_df['Pearson_r'].mean():.3f}")
print(f"Median correlation: {corr_df['Pearson_r'].median():.3f}")
print(f"Min correlation: {corr_df['Pearson_r'].min():.3f} ({corr_df.loc[corr_df['Pearson_r'].idxmin(), 'Pathway']})")
print(f"Max correlation: {corr_df['Pearson_r'].max():.3f} ({corr_df.loc[corr_df['Pearson_r'].idxmax(), 'Pathway']})")
print(f"Pathways with r > 0.7: {(corr_df['Pearson_r'] > 0.7).sum()}/{len(corr_df)}")
print(f"Pathways with r > 0.8: {(corr_df['Pearson_r'] > 0.8).sum()}/{len(corr_df)}")

# Create scatter plots for main pathways
print("\n" + "="*80)
print("GENERATING SCATTER PLOTS")
print("="*80)

main_pathways = ['Glycolysis', 'Oxidative_PPP', 'TCA_Cycle',
                 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis', 'Pyruvate_Dehydrogenase']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, pathway in enumerate(main_pathways):
    ax = axes[idx]

    if pathway in pathway_pairs:
        rate_col = pathway_pairs[pathway]

        valid_mask = (
            np.isfinite(merged_df[pathway]) &
            np.isfinite(merged_df[rate_col]) &
            (merged_df[pathway] > 0) &
            (merged_df[rate_col] > 0)
        )

        x = merged_df.loc[valid_mask, pathway]
        y = merged_df.loc[valid_mask, rate_col]

        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.1f}x + {z[1]:.1f}')

        # Correlation
        r, pval = pearsonr(x, y)

        # Labels
        ax.set_xlabel(f'Geometric Mean v2 (0-1 scale)', fontsize=10)
        ax.set_ylabel(f'Rate-Limiting (TPM)', fontsize=10)
        ax.set_title(f'{pathway}\nr = {r:.3f}, p < 0.001', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('method_comparison_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved scatter plots to method_comparison_scatter_plots.png")
plt.close()

# Create correlation heatmap
print("\nGenerating correlation heatmap...")

# Build correlation matrix
pathways_to_plot = [p for p in main_pathways if p in pathway_pairs]
corr_matrix = np.zeros((len(pathways_to_plot), len(pathways_to_plot)))

for i, p1 in enumerate(pathways_to_plot):
    for j, p2 in enumerate(pathways_to_plot):
        rate_col1 = pathway_pairs[p1]
        rate_col2 = pathway_pairs[p2]

        valid_mask = (
            np.isfinite(merged_df[p1]) &
            np.isfinite(merged_df[rate_col2]) &
            (merged_df[p1] > 0) &
            (merged_df[rate_col2] > 0)
        )

        if valid_mask.sum() > 3:
            r, _ = pearsonr(merged_df.loc[valid_mask, p1],
                          merged_df.loc[valid_mask, rate_col2])
            corr_matrix[i, j] = r

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            xticklabels=[p.replace('_', ' ') for p in pathways_to_plot],
            yticklabels=[p.replace('_', ' ') for p in pathways_to_plot],
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            center=0.5,
            cbar_kws={'label': 'Pearson r'})
plt.title('Geometric Mean v2 vs Rate-Limiting Method\nCross-Method Correlations',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Rate-Limiting Method', fontsize=12)
plt.ylabel('Geometric Mean v2 Method', fontsize=12)
plt.tight_layout()
plt.savefig('method_comparison_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved heatmap to method_comparison_heatmap.png")
plt.close()

# Create comparison barplot showing mean values (normalized)
print("\nGenerating mean comparison plot...")

comparison_data = []
for pathway in main_pathways:
    if pathway in pathway_pairs:
        rate_col = pathway_pairs[pathway]

        # Geometric mean v2 (already 0-1 scale)
        geom_mean = merged_df[pathway].mean()

        # Rate-limiting (normalize to 0-1 for comparison)
        rate_values = merged_df[rate_col]
        rate_normalized = (rate_values - rate_values.min()) / (rate_values.max() - rate_values.min())
        rate_mean = rate_normalized.mean()

        comparison_data.append({
            'Pathway': pathway.replace('_', ' '),
            'Geometric Mean v2': geom_mean,
            'Rate-Limiting (normalized)': rate_mean
        })

comp_df = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.35

ax.bar(x - width/2, comp_df['Geometric Mean v2'], width,
       label='Geometric Mean v2', alpha=0.8, color='steelblue')
ax.bar(x + width/2, comp_df['Rate-Limiting (normalized)'], width,
       label='Rate-Limiting (normalized)', alpha=0.8, color='coral')

ax.set_xlabel('Pathway', fontsize=12)
ax.set_ylabel('Mean Capacity (0-1 scale)', fontsize=12)
ax.set_title('Mean Pathway Capacity Comparison\nGeometric Mean v2 vs Rate-Limiting Method',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comp_df['Pathway'], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('method_comparison_means.png', dpi=300, bbox_inches='tight')
print("✓ Saved mean comparison to method_comparison_means.png")
plt.close()

# Create scale comparison plot
print("\nGenerating scale comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

pathways_sorted = corr_df.sort_values('Scale_Ratio')
colors = ['green' if r > 0.7 else 'orange' if r > 0.5 else 'red'
          for r in pathways_sorted['Pearson_r']]

ax.barh(range(len(pathways_sorted)), pathways_sorted['Scale_Ratio'], color=colors, alpha=0.7)
ax.set_yticks(range(len(pathways_sorted)))
ax.set_yticklabels(pathways_sorted['Pathway'].str.replace('_', ' '))
ax.set_xlabel('Scale Ratio (Rate-Limiting TPM / Geometric Mean v2)', fontsize=12)
ax.set_title('Scale Differences Between Methods\nColor = Correlation Strength',
             fontsize=14, fontweight='bold')
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='1:1 ratio')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add correlation values as text
for i, (idx, row) in enumerate(pathways_sorted.iterrows()):
    ax.text(row['Scale_Ratio'] + 10, i, f"r={row['Pearson_r']:.2f}",
            va='center', fontsize=8)

plt.tight_layout()
plt.savefig('method_comparison_scale_ratios.png', dpi=300, bbox_inches='tight')
print("✓ Saved scale comparison to method_comparison_scale_ratios.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nOutput files generated:")
print("  1. method_comparison_correlations.csv - Correlation statistics")
print("  2. method_comparison_scatter_plots.png - 6-panel scatter plots")
print("  3. method_comparison_heatmap.png - Cross-method correlation heatmap")
print("  4. method_comparison_means.png - Mean capacity comparison")
print("  5. method_comparison_scale_ratios.png - Scale ratio comparison")
print("\n" + "="*80)
