#!/usr/bin/env python3
"""
Geometric Mean Pathway Flux Estimation

New method for calculating pathway flux capacity:
1. For each gene, scale expression to the mean of top 5 samples (ceiling at 1.0)
2. For each step, compute reaction rate as sum of all isozymes
3. For each pathway, compute flux as geometric mean of all steps

This complements the rate-limiting enzyme approach by considering all genes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
print("Loading transcriptomics data...")
data = pd.read_csv('transcriptomics_data.csv', low_memory=False)
print(f"Loaded data: {data.shape[0]} genes × {data.shape[1]} columns")

# Get sample columns (exclude metadata)
sample_cols = [col for col in data.columns if col not in ['gene_id', 'gene_name', 'entrez', 'gene_type']]
print(f"Found {len(sample_cols)} samples")

# Convert sample columns to numeric
print("Converting expression values to numeric...")
for col in sample_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# ============================================================================
# PATHWAY DEFINITIONS - Organized by Steps
# ============================================================================

PATHWAY_STEPS = {
    'Glycolysis': {
        'description': 'Glucose → Pyruvate',
        'steps': {
            'Step1_Hexokinase': ['Hk1', 'Hk2', 'Hk3', 'Hk4'],  # Glucose → G6P
            'Step2_GPI': ['Gpi1'],  # G6P → F6P
            'Step3_PFK': ['Pfkl', 'Pfkm', 'Pfkp'],  # F6P → F16BP (rate-limiting)
            'Step4_Aldolase': ['Aldoa', 'Aldob', 'Aldoc'],  # F16BP → DHAP + G3P
            'Step5_TPI': ['Tpi1'],  # DHAP ⇌ G3P
            'Step6_GAPDH': ['Gapdh', 'Gapdhs'],  # G3P → 13BPG
            'Step7_PGK': ['Pgk1', 'Pgk2'],  # 13BPG → 3PG
            'Step8_PGAM': ['Pgam1', 'Pgam2'],  # 3PG → 2PG
            'Step9_Enolase': ['Eno1', 'Eno2', 'Eno3', 'Eno4'],  # 2PG → PEP
            'Step10_PK': ['Pklr', 'Pkm']  # PEP → Pyruvate
        }
    },

    'Oxidative_PPP': {
        'description': 'G6P → Ribulose-5-P + 2 NADPH',
        'steps': {
            'Step1_G6PD': ['G6pdx', 'G6pd2'],  # G6P → 6PG (rate-limiting)
            'Step2_6PGL': ['Pgls'],  # 6PGL → 6PG
            'Step3_6PGD': ['Pgd']  # 6PG → Ru5P
        }
    },

    'Non_oxidative_PPP': {
        'description': 'Pentose phosphate interconversions',
        'steps': {
            'Step1_TKT': ['Tkt'],  # Transketolase
            'Step2_TALDO': ['Taldo1']  # Transaldolase
        }
    },

    'TCA_Cycle': {
        'description': 'Acetyl-CoA → 2 CO2 + 3 NADH + FADH2 + GTP',
        'steps': {
            'Step1_CS': ['Cs'],  # Acetyl-CoA + OAA → Citrate
            'Step2_Aconitase': ['Aco1', 'Aco2'],  # Citrate → Isocitrate
            'Step3_IDH': ['Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g'],  # Isocitrate → α-KG
            'Step4_OGDH': ['Ogdh', 'Dlst'],  # α-KG → Succinyl-CoA
            'Step5_SCS': ['Sucla2', 'Suclg1', 'Suclg2'],  # Succinyl-CoA → Succinate
            'Step6_SDH': ['Sdha', 'Sdhb', 'Sdhc', 'Sdhd'],  # Succinate → Fumarate
            'Step7_Fumarase': ['Fh1'],  # Fumarate → Malate
            'Step8_MDH': ['Mdh1', 'Mdh2']  # Malate → OAA
        }
    },

    'Pyruvate_Dehydrogenase': {
        'description': 'Pyruvate → Acetyl-CoA (Carbohydrate → TCA)',
        'steps': {
            'Step1_E1': ['Pdha1', 'Pdhb'],  # E1 component
            'Step2_E2': ['Dlat'],  # E2 component
            'Step3_E3': ['Dld']  # E3 component
        }
    },

    'Fatty_Acid_Oxidation': {
        'description': 'Fatty acids → Acetyl-CoA (FA → TCA)',
        'steps': {
            'Step1_CPT1': ['Cpt1a', 'Cpt1b', 'Cpt1c'],  # Mitochondrial import
            'Step2_CPT2': ['Cpt2'],  # Inner membrane
            'Step3_ACAD': ['Acadvl', 'Acadm', 'Acads', 'Acadl'],  # Acyl-CoA dehydrogenase
            'Step4_Trifunctional': ['Hadha', 'Hadhb'],  # 3 steps of β-oxidation
            'Step5_Enoyl': ['Echs1']  # Enoyl-CoA hydratase
        }
    },

    'Fatty_Acid_Synthesis': {
        'description': 'Acetyl-CoA → Palmitate',
        'steps': {
            'Step1_ACC': ['Acaca', 'Acacb'],  # Acetyl-CoA → Malonyl-CoA
            'Step2_FAS': ['Fasn'],  # Fatty acid synthase complex
            'Step3_SCD': ['Scd1', 'Scd2', 'Scd3', 'Scd4']  # Desaturation
        }
    },

    'Malic_Enzyme': {
        'description': 'Malate → Pyruvate + NADPH',
        'steps': {
            'Step1_ME': ['Me1', 'Me2', 'Me3']  # Single-step pathway
        }
    },

    'IDH_NADPH': {
        'description': 'Isocitrate → α-KG + NADPH',
        'steps': {
            'Step1_IDH': ['Idh1', 'Idh2']  # NADP-dependent IDH
        }
    },

    'Glutaminolysis': {
        'description': 'Glutamine → α-KG (AA → TCA)',
        'steps': {
            'Step1_GLS': ['Gls', 'Gls2'],  # Glutamine → Glutamate
            'Step2_GLUD': ['Glud1', 'Glud2'],  # Glutamate → α-KG
            'Step3_GOT': ['Got1', 'Got2']  # Alternative transamination
        }
    }
}

# Additional amino acid catabolism pathways (simplified to key steps)
AA_CATABOLISM_STEPS = {
    'Alanine_catabolism': {
        'description': 'Alanine → Pyruvate',
        'steps': {
            'Step1_ALT': ['Gpt', 'Gpt2']  # Alanine aminotransferase
        }
    },

    'Aspartate_catabolism': {
        'description': 'Aspartate → Oxaloacetate',
        'steps': {
            'Step1_GOT': ['Got1', 'Got2']  # Aspartate aminotransferase
        }
    },

    'Leucine_catabolism': {
        'description': 'Leucine → Acetyl-CoA + Acetoacetate',
        'steps': {
            'Step1_BCAT': ['Bcat1', 'Bcat2'],  # Branched-chain aminotransferase
            'Step2_BCKDH': ['Bckdha', 'Bckdhb'],  # Branched-chain α-keto acid dehydrogenase
            'Step3_IVD': ['Ivd'],  # Isovaleryl-CoA dehydrogenase
            'Step4_MCCC': ['Mccc1', 'Mccc2']  # Methylcrotonyl-CoA carboxylase
        }
    },

    'Isoleucine_catabolism': {
        'description': 'Isoleucine → Acetyl-CoA + Succinyl-CoA',
        'steps': {
            'Step1_BCAT': ['Bcat1', 'Bcat2'],
            'Step2_BCKDH': ['Bckdha', 'Bckdhb'],
            'Step3_ACAD': ['Acadsb']
        }
    },

    'Valine_catabolism': {
        'description': 'Valine → Succinyl-CoA',
        'steps': {
            'Step1_BCAT': ['Bcat1', 'Bcat2'],
            'Step2_BCKDH': ['Bckdha', 'Bckdhb'],
            'Step3_Hibadh': ['Hibadh'],
            'Step4_Hibch': ['Hibch']
        }
    },

    'Serine_catabolism': {
        'description': 'Serine → Pyruvate',
        'steps': {
            'Step1_SDH': ['Sdsl'],  # Serine dehydratase
            'Step2_SHMT': ['Shmt1', 'Shmt2']  # Serine hydroxymethyltransferase
        }
    },

    'Glycine_catabolism': {
        'description': 'Glycine → CO2 + NH3 + NADH',
        'steps': {
            'Step1_GCS': ['Gcsh'],  # Glycine cleavage system
            'Step2_AMT': ['Amt']  # Aminomethyltransferase
        }
    },

    'Threonine_catabolism': {
        'description': 'Threonine → Succinyl-CoA',
        'steps': {
            'Step1_TDH': ['Tdh'],  # Threonine dehydrogenase
            'Step2_SDH': ['Sdsl']  # Serine dehydratase
        }
    },

    'Methionine_catabolism': {
        'description': 'Methionine → Succinyl-CoA',
        'steps': {
            'Step1_MAT': ['Mat1a', 'Mat2a', 'Mat2b'],  # Methionine adenosyltransferase
            'Step2_CBS': ['Cbs'],  # Cystathionine β-synthase
            'Step3_CTH': ['Cth']  # Cystathionine γ-lyase
        }
    },

    'Proline_catabolism': {
        'description': 'Proline → α-KG',
        'steps': {
            'Step1_PRODH': ['Prodh', 'Prodh2'],  # Proline dehydrogenase
            'Step2_P5CDH': ['Aldh4a1']  # Δ1-pyrroline-5-carboxylate dehydrogenase
        }
    },

    'Arginine_catabolism': {
        'description': 'Arginine → α-KG',
        'steps': {
            'Step1_ARG': ['Arg1', 'Arg2'],  # Arginase
            'Step2_OAT': ['Oat']  # Ornithine aminotransferase
        }
    },

    'Histidine_catabolism': {
        'description': 'Histidine → α-KG',
        'steps': {
            'Step1_HAL': ['Hal'],  # Histidase
            'Step2_UROC1': ['Uroc1'],  # Urocanate hydratase
            'Step3_FTCD': ['Ftcd']  # Formiminotransferase cyclodeaminase
        }
    },

    'Lysine_catabolism': {
        'description': 'Lysine → Acetyl-CoA',
        'steps': {
            'Step1_AASS': ['Aass'],  # α-aminoadipic semialdehyde synthase
            'Step2_AASDH': ['Aasdh'],  # α-aminoadipic semialdehyde dehydrogenase
            'Step3_DHTKD1': ['Dhtkd1']  # Dehydrogenase E1 and transketolase domain-containing 1
        }
    },

    'Phenylalanine_catabolism': {
        'description': 'Phenylalanine → Fumarate + Acetoacetate',
        'steps': {
            'Step1_PAH': ['Pah'],  # Phenylalanine hydroxylase
            'Step2_TAT': ['Tat'],  # Tyrosine aminotransferase
            'Step3_HPD': ['Hpd'],  # 4-hydroxyphenylpyruvate dioxygenase
            'Step4_HGD': ['Hgd']  # Homogentisate 1,2-dioxygenase
        }
    },

    'Tyrosine_catabolism': {
        'description': 'Tyrosine → Fumarate + Acetoacetate',
        'steps': {
            'Step1_TAT': ['Tat'],
            'Step2_HPD': ['Hpd'],
            'Step3_HGD': ['Hgd']
        }
    },

    'Tryptophan_catabolism': {
        'description': 'Tryptophan → Acetyl-CoA',
        'steps': {
            'Step1_TDO': ['Tdo2'],  # Tryptophan 2,3-dioxygenase
            'Step2_KYNU': ['Kynu'],  # Kynureninase
            'Step3_HAAO': ['Haao']  # 3-hydroxyanthranilate 3,4-dioxygenase
        }
    },

    'Cysteine_catabolism': {
        'description': 'Cysteine → Pyruvate',
        'steps': {
            'Step1_CDO1': ['Cdo1'],  # Cysteine dioxygenase
            'Step2_GOT': ['Got1', 'Got2']  # Transamination
        }
    },

    'Asparagine_catabolism': {
        'description': 'Asparagine → Oxaloacetate',
        'steps': {
            'Step1_ASNS': ['Asns'],  # Asparagine synthetase (reversible)
            'Step2_GOT': ['Got1', 'Got2']  # Aspartate pathway
        }
    }
}

# Combine all pathways
ALL_PATHWAYS = {**PATHWAY_STEPS, **AA_CATABOLISM_STEPS}

# ============================================================================
# STEP 1: Scale each gene expression to mean of top 5 samples (ceiling at 1.0)
# ============================================================================

def scale_gene_expression(data, sample_cols):
    """
    For each gene, scale expression to the mean of top 5 samples.
    Floor at 0.05, Ceiling at 1.0.

    Returns: DataFrame with scaled expression (vectorized for speed)
    """
    print("\nStep 1: Scaling gene expression to top 5 sample mean (Floor: 0.05, Ceiling: 1.0)...")

    # Extract expression matrix
    expr_matrix = data[sample_cols].values

    # For each gene (row), find mean of top 5 samples
    # Sort each row descending and take mean of first 5
    top5_means = np.mean(np.sort(expr_matrix, axis=1)[:, -5:], axis=1)

    # Avoid division by zero
    top5_means[top5_means == 0] = 1.0

    # Scale each row by its top5 mean
    scaled_matrix = expr_matrix / top5_means[:, np.newaxis]

    # Apply ceiling at 1.0 and floor at 0.05
    scaled_matrix = np.minimum(scaled_matrix, 1.0)
    scaled_matrix = np.maximum(scaled_matrix, 0.05)

    # Create new dataframe with scaled values
    scaled_data = data.copy()
    scaled_data[sample_cols] = scaled_matrix

    print(f"✓ Scaled {len(data)} genes")
    return scaled_data

# ============================================================================
# STEP 2: Calculate pathway flux using geometric mean
# ============================================================================

def calculate_geometric_mean_flux(sample, pathway_steps_dict, scaled_data, sample_cols):
    """
    Calculate pathway flux capacity using geometric mean method v2:
    1. For each step, MAX of isozyme expressions (dominant isoform)
    2. For each pathway, geometric mean of all step rates
    3. TPI excluded from Glycolysis pathway

    Returns: Dictionary of pathway capacities
    """
    capacity = {}

    for pathway_name, pathway_info in pathway_steps_dict.items():
        steps = pathway_info['steps']
        step_rates = []

        # For each step in the pathway
        for step_name, genes in steps.items():
            # Skip TPI for Glycolysis pathway
            if pathway_name == 'Glycolysis' and 'TPI' in step_name.upper():
                continue

            isozyme_expressions = []

            # Get expression for each isozyme
            for gene_name in genes:
                gene_row = scaled_data[scaled_data['gene_name'].str.upper() == gene_name.upper()]
                if len(gene_row) > 0 and sample in sample_cols:
                    expr = gene_row[sample].values[0]
                    if not np.isnan(expr):
                        isozyme_expressions.append(expr)

            # Step rate = MAX of all isozymes (v2 change)
            if len(isozyme_expressions) > 0:
                step_rate = np.max(isozyme_expressions)
                step_rates.append(step_rate)

        # Pathway flux = geometric mean of all steps
        if len(step_rates) > 0 and all(r > 0 for r in step_rates):
            # Geometric mean: (product of all values)^(1/n)
            pathway_flux = np.exp(np.mean(np.log(step_rates)))
            capacity[pathway_name] = pathway_flux
        elif len(step_rates) > 0:
            # If any step is zero, geometric mean is zero
            capacity[pathway_name] = 0.0
        else:
            capacity[pathway_name] = 0.0

    return capacity

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("GEOMETRIC MEAN PATHWAY FLUX ESTIMATION")
print("="*80)

# Scale gene expression
scaled_data = scale_gene_expression(data, sample_cols)

# Calculate pathway capacities for all samples
print("\nStep 2: Calculating pathway flux using geometric mean of steps...")
results = []

for i, sample in enumerate(sample_cols):
    if (i + 1) % 50 == 0:
        print(f"Processing sample {i+1}/{len(sample_cols)}...")

    pathway_capacity = calculate_geometric_mean_flux(sample, ALL_PATHWAYS, scaled_data, sample_cols)

    result = {'Sample': sample}
    result.update(pathway_capacity)
    results.append(result)

results_df = pd.DataFrame(results)

# Save results
output_file = 'geometric_mean_pathway_fluxes.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Saved geometric mean flux results to {output_file}")
print(f"  Shape: {results_df.shape}")

# ============================================================================
# COMPARISON WITH RATE-LIMITING ENZYME METHOD
# ============================================================================

print("\n" + "="*80)
print("COMPARING WITH RATE-LIMITING ENZYME METHOD")
print("="*80)

# Load rate-limiting enzyme results (if available)
try:
    rate_lim_df = pd.read_csv('all_samples_calibrated_metabolism.csv')

    # Common pathways for comparison
    common_pathways = [
        'Glycolysis', 'Oxidative_PPP', 'TCA_Cycle',
        'Pyruvate_Dehydrogenase', 'Fatty_Acid_Oxidation',
        'Fatty_Acid_Synthesis', 'Glutaminolysis'
    ]

    # Map column names (rate-limiting uses different naming)
    column_map = {
        'Glycolysis': 'Glycolysis_capacity',
        'Oxidative_PPP': 'Oxidative_PPP_capacity',
        'TCA_Cycle': 'TCA_Cycle_capacity',
        'Pyruvate_Dehydrogenase': 'Pyruvate_Dehydrogenase_capacity',
        'Fatty_Acid_Oxidation': 'Fatty_Acid_Oxidation_capacity',
        'Fatty_Acid_Synthesis': 'Fatty_Acid_Synthesis_capacity',
        'Glutaminolysis': 'Glutaminolysis_capacity'
    }

    # Merge datasets on Sample
    merged = results_df.merge(rate_lim_df[['Sample'] + list(column_map.values())],
                              on='Sample', how='inner')

    # Calculate correlations
    print("\nCorrelations between Geometric Mean and Rate-Limiting methods:")
    print("-" * 70)

    for geom_name, rate_lim_name in column_map.items():
        if geom_name in merged.columns and rate_lim_name in merged.columns:
            # Remove zeros for correlation
            mask = (merged[geom_name] > 0) & (merged[rate_lim_name] > 0)
            if mask.sum() > 10:
                corr, pval = stats.pearsonr(merged.loc[mask, geom_name],
                                           merged.loc[mask, rate_lim_name])
                print(f"{geom_name:30s}: r = {corr:.3f}, p = {pval:.2e}")

    # Save comparison
    comparison_file = 'method_comparison_geometric_vs_ratelimiting.csv'
    comparison_cols = ['Sample'] + common_pathways + [column_map[p] for p in common_pathways if p in column_map]
    merged[comparison_cols].to_csv(comparison_file, index=False)
    print(f"\n✓ Saved method comparison to {comparison_file}")

except FileNotFoundError:
    print("\nRate-limiting enzyme results not found. Skipping comparison.")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS - GEOMETRIC MEAN METHOD")
print("="*80)

# Calculate statistics for main pathways
main_pathways = ['Glycolysis', 'Oxidative_PPP', 'Non_oxidative_PPP', 'TCA_Cycle',
                 'Pyruvate_Dehydrogenase', 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis',
                 'Malic_Enzyme', 'IDH_NADPH', 'Glutaminolysis']

summary_stats = []
for pathway in main_pathways:
    if pathway in results_df.columns:
        values = results_df[pathway].values
        non_zero = values[values > 0]

        summary_stats.append({
            'Pathway': pathway,
            'Mean': np.mean(values),
            'Median': np.median(values),
            'SD': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Non_zero_samples': len(non_zero),
            'Percent_non_zero': 100 * len(non_zero) / len(values)
        })

summary_df = pd.DataFrame(summary_stats)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('geometric_mean_summary_statistics.csv', index=False)

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Geometric Mean Pathway Flux Estimation - Distribution Across Samples',
             fontsize=16, fontweight='bold')

plot_pathways = main_pathways[:9]

for idx, pathway in enumerate(plot_pathways):
    ax = axes[idx // 3, idx % 3]

    if pathway in results_df.columns:
        values = results_df[pathway].values
        non_zero = values[values > 0]

        if len(non_zero) > 0:
            ax.hist(non_zero, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(non_zero), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(non_zero):.3f}')
            ax.axvline(np.median(non_zero), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(non_zero):.3f}')

        ax.set_xlabel('Scaled Flux Capacity', fontsize=10)
        ax.set_ylabel('Number of Samples', fontsize=10)
        ax.set_title(pathway.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('geometric_mean_flux_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved distribution plot to geometric_mean_flux_distributions.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  1. geometric_mean_pathway_fluxes.csv - All pathway fluxes for all samples")
print(f"  2. geometric_mean_summary_statistics.csv - Summary statistics")
print(f"  3. geometric_mean_flux_distributions.png - Distribution plots")
if 'comparison_file' in locals():
    print(f"  4. {comparison_file} - Method comparison")
