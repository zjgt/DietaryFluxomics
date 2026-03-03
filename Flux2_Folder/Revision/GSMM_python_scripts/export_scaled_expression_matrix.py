#!/usr/bin/env python3
"""
Export Scaled Gene Expression Matrix for Geometric Mean v2
Organized by tissue and gender groups
"""

import pandas as pd
import numpy as np

print("="*80)
print("EXPORTING SCALED GENE EXPRESSION MATRIX")
print("="*80)

# Load Sample Info to get correct sample list
print("\nLoading Sample_Info.csv...")
sample_info = pd.read_csv('Sample_Info.csv')
valid_samples = sample_info['Sample'].tolist()
print(f"Found {len(valid_samples)} valid samples")

# Load transcriptomics data
print("\nLoading transcriptomics data...")
data = pd.read_csv('transcriptomics_data.csv', low_memory=False)
print(f"Loaded data: {data.shape[0]} genes × {data.shape[1]} columns")

# Filter to only valid samples
sample_cols = [col for col in data.columns if col in valid_samples]
sample_cols = [col for col in sample_cols if '.' not in col]
print(f"Valid sample columns: {len(sample_cols)}")

# Convert to numeric and fill NaN
print("\nConverting expression values to numeric...")
for col in sample_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Step 1: Scale gene expression (Floor: 0.05, Ceiling: 1.0)
print("\nScaling gene expression (Floor: 0.05, Ceiling: 1.0)...")
expr_matrix = data[sample_cols].values
top5_means = np.mean(np.sort(expr_matrix, axis=1)[:, -5:], axis=1)
top5_means[top5_means == 0] = 1.0

scaled_matrix = expr_matrix / top5_means[:, np.newaxis]
scaled_matrix = np.minimum(scaled_matrix, 1.0)
scaled_matrix = np.maximum(scaled_matrix, 0.05)

# Create scaled data DataFrame
scaled_data = pd.DataFrame(scaled_matrix,
                          columns=sample_cols,
                          index=data.index)

# Add gene information
gene_cols = ['gene_name', 'description', 'gene_id']
for col in gene_cols:
    if col in data.columns:
        scaled_data.insert(0, col, data[col])

print(f"✓ Scaled {len(data)} genes × {len(sample_cols)} samples")

# Parse tissue and sex from sample names
print("\nParsing sample metadata...")
sample_metadata = []

for sample in sample_cols:
    sample_row = sample_info[sample_info['Sample'] == sample]
    if len(sample_row) > 0:
        name = sample_row['Name'].values[0]
        description = sample_row['Description'].values[0]
        if pd.notna(name):
            parts = name.split('_')
            tissue = parts[0] if len(parts) > 0 else 'Unknown'
        else:
            tissue = 'Unknown'

        if pd.notna(description):
            if description.startswith('Male'):
                sex_full = 'Male'
            elif description.startswith('Female'):
                sex_full = 'Female'
            else:
                sex_full = 'Unknown'
        else:
            sex_full = 'Unknown'
    else:
        tissue = 'Unknown'
        sex_full = 'Unknown'

    sample_metadata.append({
        'Sample': sample,
        'Tissue': tissue,
        'Sex': sex_full
    })

metadata_df = pd.DataFrame(sample_metadata)
print(f"Found {metadata_df['Tissue'].nunique()} tissues")
print(f"Sex distribution: {metadata_df['Sex'].value_counts().to_dict()}")

# Export 1: Full scaled matrix with all genes
print("\n" + "="*80)
print("EXPORT 1: Full Scaled Expression Matrix")
print("="*80)

scaled_data.to_csv('geometric_mean_v2_scaled_expression_full.csv', index=False)
print(f"✓ Saved: geometric_mean_v2_scaled_expression_full.csv")
print(f"  Shape: {scaled_data.shape}")
print(f"  Contains: All {len(data)} genes × {len(sample_cols)} samples")

# Export 2: Pathway genes only (more manageable size)
print("\n" + "="*80)
print("EXPORT 2: Pathway Genes Only (Scaled Expression)")
print("="*80)

# Define all pathway genes
PATHWAY_GENES = {
    'Glycolysis': ['Hk1', 'Hk2', 'Hk3', 'Hk4', 'Gpi1', 'Pfkl', 'Pfkm', 'Pfkp',
                   'Aldoa', 'Aldob', 'Aldoc', 'Tpi1', 'Gapdh', 'Gapdhs',
                   'Pgk1', 'Pgk2', 'Pgam1', 'Pgam2', 'Eno1', 'Eno2', 'Eno3', 'Eno4',
                   'Pklr', 'Pkm'],
    'Oxidative_PPP': ['G6pdx', 'G6pd2', 'Pgls', 'Pgd'],
    'Non_oxidative_PPP': ['Tkt', 'Taldo1'],
    'TCA_Cycle': ['Cs', 'Aco1', 'Aco2', 'Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g',
                  'Ogdh', 'Dlst', 'Sucla2', 'Suclg1', 'Suclg2',
                  'Sdha', 'Sdhb', 'Sdhc', 'Sdhd', 'Fh1', 'Mdh1', 'Mdh2'],
    'Pyruvate_Dehydrogenase': ['Pdha1', 'Pdhb', 'Dlat', 'Dld'],
    'Fatty_Acid_Oxidation': ['Cpt1a', 'Cpt1b', 'Cpt1c', 'Cpt2',
                             'Acadvl', 'Acadm', 'Acads', 'Acadl',
                             'Hadha', 'Hadhb', 'Echs1'],
    'Fatty_Acid_Synthesis': ['Acaca', 'Acacb', 'Fasn', 'Scd1', 'Scd2', 'Scd3', 'Scd4'],
    'Malic_Enzyme': ['Me1', 'Me2', 'Me3'],
    'IDH_NADPH': ['Idh1', 'Idh2'],
    'Glutaminolysis': ['Gls', 'Gls2', 'Glud1', 'Glud2', 'Got1', 'Got2'],
    'Alanine_catabolism': ['Gpt', 'Gpt2'],
    'Aspartate_catabolism': ['Got1', 'Got2'],
    'Leucine_catabolism': ['Bcat1', 'Bcat2', 'Bckdha', 'Bckdhb', 'Ivd', 'Mccc1', 'Mccc2'],
    'Isoleucine_catabolism': ['Bcat1', 'Bcat2', 'Bckdha', 'Bckdhb', 'Acadsb'],
    'Valine_catabolism': ['Bcat1', 'Bcat2', 'Bckdha', 'Bckdhb', 'Hibadh', 'Hibch'],
    'Serine_catabolism': ['Sdsl', 'Shmt1', 'Shmt2'],
    'Glycine_catabolism': ['Gcsh', 'Amt'],
    'Threonine_catabolism': ['Tdh', 'Sdsl'],
    'Methionine_catabolism': ['Mat1a', 'Mat2a', 'Mat2b', 'Cbs', 'Cth'],
    'Proline_catabolism': ['Prodh', 'Prodh2', 'Aldh4a1'],
    'Arginine_catabolism': ['Arg1', 'Arg2', 'Oat'],
    'Histidine_catabolism': ['Hal', 'Uroc1', 'Ftcd'],
    'Lysine_catabolism': ['Aass', 'Aasdh', 'Dhtkd1'],
    'Phenylalanine_catabolism': ['Pah', 'Tat', 'Hpd', 'Hgd'],
    'Tyrosine_catabolism': ['Tat', 'Hpd', 'Hgd'],
    'Tryptophan_catabolism': ['Tdo2', 'Kynu', 'Haao'],
    'Cysteine_catabolism': ['Cdo1', 'Got1', 'Got2'],
    'Asparagine_catabolism': ['Asns', 'Got1', 'Got2']
}

# Collect all unique pathway genes
all_pathway_genes = set()
for genes in PATHWAY_GENES.values():
    all_pathway_genes.update([g.upper() for g in genes])

print(f"Total unique pathway genes: {len(all_pathway_genes)}")

# Filter scaled data to pathway genes only
if 'gene_name' in scaled_data.columns:
    pathway_genes_mask = scaled_data['gene_name'].str.upper().isin(all_pathway_genes)
    scaled_pathway = scaled_data[pathway_genes_mask].copy()

    scaled_pathway.to_csv('geometric_mean_v2_scaled_expression_pathway_genes.csv', index=False)
    print(f"✓ Saved: geometric_mean_v2_scaled_expression_pathway_genes.csv")
    print(f"  Shape: {scaled_pathway.shape}")
    print(f"  Contains: {len(scaled_pathway)} pathway genes × {len(sample_cols)} samples")

# Export 3: Organized by tissue-sex groups
print("\n" + "="*80)
print("EXPORT 3: Samples Organized by Tissue-Sex Groups")
print("="*80)

# Create column order: gene info columns + samples grouped by tissue-sex
tissue_sex_groups = metadata_df.groupby(['Tissue', 'Sex'])['Sample'].apply(list).to_dict()

# Sort groups
sorted_groups = sorted(tissue_sex_groups.keys(),
                       key=lambda x: (x[0], x[1]))  # Sort by tissue, then sex

print(f"\nTissue-Sex groups found: {len(sorted_groups)}")

# Create new column order
new_col_order = []

# Add gene info columns first
for col in ['gene_name', 'description', 'gene_id']:
    if col in scaled_data.columns:
        new_col_order.append(col)

# Add samples grouped by tissue-sex
group_info = []
for tissue, sex in sorted_groups:
    samples = tissue_sex_groups[(tissue, sex)]
    new_col_order.extend(samples)
    group_info.append({
        'Tissue': tissue,
        'Sex': sex,
        'Sample_Count': len(samples),
        'Samples': ','.join(samples)
    })
    print(f"  {tissue} - {sex}: {len(samples)} samples")

# Reorder columns
scaled_data_grouped = scaled_data[new_col_order].copy()

scaled_data_grouped.to_csv('geometric_mean_v2_scaled_expression_grouped.csv', index=False)
print(f"\n✓ Saved: geometric_mean_v2_scaled_expression_grouped.csv")
print(f"  Shape: {scaled_data_grouped.shape}")
print(f"  Columns organized by tissue-sex groups")

# Save group information
group_info_df = pd.DataFrame(group_info)
group_info_df.to_csv('geometric_mean_v2_sample_groups.csv', index=False)
print(f"✓ Saved: geometric_mean_v2_sample_groups.csv")
print(f"  Contains: {len(group_info)} tissue-sex combinations")

# Export 4: Pathway genes grouped by tissue-sex
print("\n" + "="*80)
print("EXPORT 4: Pathway Genes Grouped by Tissue-Sex")
print("="*80)

if 'gene_name' in scaled_data.columns:
    scaled_pathway_grouped = scaled_data_grouped[pathway_genes_mask].copy()

    scaled_pathway_grouped.to_csv('geometric_mean_v2_scaled_pathway_genes_grouped.csv',
                                   index=False)
    print(f"✓ Saved: geometric_mean_v2_scaled_pathway_genes_grouped.csv")
    print(f"  Shape: {scaled_pathway_grouped.shape}")
    print(f"  Contains: Pathway genes only, organized by tissue-sex groups")

# Export 5: Summary of scaling parameters
print("\n" + "="*80)
print("EXPORT 5: Scaling Parameters Summary")
print("="*80)

# Calculate scaling statistics for each gene
scaling_stats = []

for idx in range(len(data)):
    gene_name = data.loc[idx, 'gene_name'] if 'gene_name' in data.columns else f"Gene_{idx}"

    original_values = expr_matrix[idx, :]
    scaled_values = scaled_matrix[idx, :]
    top5_mean = top5_means[idx]

    scaling_stats.append({
        'gene_name': gene_name,
        'top5_mean': top5_mean,
        'original_mean': np.mean(original_values),
        'original_max': np.max(original_values),
        'scaled_mean': np.mean(scaled_values),
        'scaled_median': np.median(scaled_values),
        'scaled_min': np.min(scaled_values),
        'scaled_max': np.max(scaled_values),
        'percent_at_floor': np.mean(scaled_values == 0.05) * 100,
        'percent_at_ceiling': np.mean(scaled_values == 1.0) * 100
    })

scaling_stats_df = pd.DataFrame(scaling_stats)
scaling_stats_df.to_csv('geometric_mean_v2_scaling_parameters.csv', index=False)
print(f"✓ Saved: geometric_mean_v2_scaling_parameters.csv")
print(f"  Shape: {scaling_stats_df.shape}")
print(f"  Contains: Scaling statistics for each gene")

# Summary statistics
print("\n" + "="*80)
print("SCALING SUMMARY")
print("="*80)
print(f"Total genes scaled: {len(data)}")
print(f"Total samples: {len(sample_cols)}")
print(f"Floor value: 0.05")
print(f"Ceiling value: 1.0")
print(f"\nGenes at floor (>50% samples): {(scaling_stats_df['percent_at_floor'] > 50).sum()}")
print(f"Genes at ceiling (>1% samples): {(scaling_stats_df['percent_at_ceiling'] > 1).sum()}")
print(f"\nMean scaled expression: {scaling_stats_df['scaled_mean'].mean():.3f}")
print(f"Median scaled expression: {scaling_stats_df['scaled_median'].median():.3f}")

print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  1. geometric_mean_v2_scaled_expression_full.csv")
print("     - All genes, original sample order")
print("  2. geometric_mean_v2_scaled_expression_pathway_genes.csv")
print("     - Pathway genes only, original sample order")
print("  3. geometric_mean_v2_scaled_expression_grouped.csv")
print("     - All genes, grouped by tissue-sex")
print("  4. geometric_mean_v2_scaled_pathway_genes_grouped.csv")
print("     - Pathway genes only, grouped by tissue-sex")
print("  5. geometric_mean_v2_sample_groups.csv")
print("     - Tissue-sex group information")
print("  6. geometric_mean_v2_scaling_parameters.csv")
print("     - Scaling statistics for each gene")
print("\n" + "="*80)
