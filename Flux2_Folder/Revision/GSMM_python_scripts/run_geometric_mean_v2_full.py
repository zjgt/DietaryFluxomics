#!/usr/bin/env python3
"""
Geometric Mean v2 - Full Dataset Analysis
Uses Sample_Info.csv to get correct sample list
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Sample Info to get correct sample list
print("Loading Sample_Info.csv...")
sample_info = pd.read_csv('Sample_Info.csv')
valid_samples = sample_info['Sample'].tolist()
print(f"Found {len(valid_samples)} valid samples")

# Load transcriptomics data
print("\nLoading transcriptomics data...")
data = pd.read_csv('transcriptomics_data.csv', low_memory=False)
print(f"Loaded data: {data.shape[0]} genes × {data.shape[1]} columns")

# Filter to only valid samples (exclude .1 duplicates)
sample_cols = [col for col in data.columns if col in valid_samples]
print(f"Found {len(sample_cols)} valid samples in expression data")

# Additional check: remove any columns with dots or that aren't in Sample_Info
sample_cols = [col for col in sample_cols if '.' not in col]
print(f"After filtering duplicates: {len(sample_cols)} samples")

# Convert to numeric and fill NaN
print("Converting expression values to numeric...")
for col in sample_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Pathway definitions with steps
GLYCOLYSIS_STEPS = {
    'Step1_HK': ['Hk1', 'Hk2', 'Hk3', 'Hk4'],
    'Step2_GPI': ['Gpi1'],
    'Step3_PFK': ['Pfkl', 'Pfkm', 'Pfkp'],
    'Step4_Aldolase': ['Aldoa', 'Aldob', 'Aldoc'],
    'Step5_TPI': ['Tpi1'],  # Will be excluded
    'Step6_GAPDH': ['Gapdh', 'Gapdhs'],
    'Step7_PGK': ['Pgk1', 'Pgk2'],
    'Step8_PGAM': ['Pgam1', 'Pgam2'],
    'Step9_Enolase': ['Eno1', 'Eno2', 'Eno3', 'Eno4'],
    'Step10_PK': ['Pklr', 'Pkm']
}

ALL_PATHWAY_STEPS = {
    'Glycolysis': {'steps': GLYCOLYSIS_STEPS},
    'Oxidative_PPP': {'steps': {
        'Step1_G6PD': ['G6pdx', 'G6pd2'],
        'Step2_6PGL': ['Pgls'],
        'Step3_6PGD': ['Pgd']
    }},
    'Non_oxidative_PPP': {'steps': {
        'Step1_TKT': ['Tkt'],
        'Step2_TALDO': ['Taldo1']
    }},
    'TCA_Cycle': {'steps': {
        'Step1_CS': ['Cs'],
        'Step2_Aconitase': ['Aco1', 'Aco2'],
        'Step3_IDH': ['Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g'],
        'Step4_OGDH': ['Ogdh', 'Dlst'],
        'Step5_SCS': ['Sucla2', 'Suclg1', 'Suclg2'],
        'Step6_SDH': ['Sdha', 'Sdhb', 'Sdhc', 'Sdhd'],
        'Step7_Fumarase': ['Fh1'],
        'Step8_MDH': ['Mdh1', 'Mdh2']
    }},
    'Pyruvate_Dehydrogenase': {'steps': {
        'Step1_E1': ['Pdha1', 'Pdhb'],
        'Step2_E2': ['Dlat'],
        'Step3_E3': ['Dld']
    }},
    'Fatty_Acid_Oxidation': {'steps': {
        'Step1_CPT1': ['Cpt1a', 'Cpt1b', 'Cpt1c'],
        'Step2_CPT2': ['Cpt2'],
        'Step3_ACAD': ['Acadvl', 'Acadm', 'Acads', 'Acadl'],
        'Step4_Trifunctional': ['Hadha', 'Hadhb'],
        'Step5_Enoyl': ['Echs1']
    }},
    'Fatty_Acid_Synthesis': {'steps': {
        'Step1_ACC': ['Acaca', 'Acacb'],
        'Step2_FAS': ['Fasn'],
        'Step3_SCD': ['Scd1', 'Scd2', 'Scd3', 'Scd4']
    }},
    'Malic_Enzyme': {'steps': {
        'Step1_ME': ['Me1', 'Me2', 'Me3']
    }},
    'IDH_NADPH': {'steps': {
        'Step1_IDH': ['Idh1', 'Idh2']
    }},
    'Lactate_Dehydrogenase': {'steps': {
        'Step1_LDH': ['Ldha', 'Ldhb', 'Ldhc', 'Ldhd']
    }},
    'Glutaminolysis': {'steps': {
        'Step1_GLS': ['Gls', 'Gls2'],
        'Step2_GLUD': ['Glud1', 'Glud2'],
        'Step3_GOT': ['Got1', 'Got2']
    }}
}

# Add amino acid pathways
AA_PATHWAYS = {
    'Alanine_catabolism': {'steps': {'Step1_ALT': ['Gpt', 'Gpt2']}},
    'Aspartate_catabolism': {'steps': {'Step1_GOT': ['Got1', 'Got2']}},
    'Leucine_catabolism': {'steps': {
        'Step1_BCAT': ['Bcat1', 'Bcat2'],
        'Step2_BCKDH': ['Bckdha', 'Bckdhb'],
        'Step3_IVD': ['Ivd'],
        'Step4_MCCC': ['Mccc1', 'Mccc2']
    }},
    'Isoleucine_catabolism': {'steps': {
        'Step1_BCAT': ['Bcat1', 'Bcat2'],
        'Step2_BCKDH': ['Bckdha', 'Bckdhb'],
        'Step3_ACAD': ['Acadsb']
    }},
    'Valine_catabolism': {'steps': {
        'Step1_BCAT': ['Bcat1', 'Bcat2'],
        'Step2_BCKDH': ['Bckdha', 'Bckdhb'],
        'Step3_Hibadh': ['Hibadh'],
        'Step4_Hibch': ['Hibch']
    }},
    'Serine_catabolism': {'steps': {
        'Step1_SDH': ['Sdsl'],
        'Step2_SHMT': ['Shmt1', 'Shmt2']
    }},
    'Glycine_catabolism': {'steps': {
        'Step1_GCS': ['Gcsh'],
        'Step2_AMT': ['Amt']
    }},
    'Threonine_catabolism': {'steps': {
        'Step1_TDH': ['Tdh'],
        'Step2_SDH': ['Sdsl']
    }},
    'Methionine_catabolism': {'steps': {
        'Step1_MAT': ['Mat1a', 'Mat2a', 'Mat2b'],
        'Step2_CBS': ['Cbs'],
        'Step3_CTH': ['Cth']
    }},
    'Proline_catabolism': {'steps': {
        'Step1_PRODH': ['Prodh', 'Prodh2'],
        'Step2_P5CDH': ['Aldh4a1']
    }},
    'Arginine_catabolism': {'steps': {
        'Step1_ARG': ['Arg1', 'Arg2'],
        'Step2_OAT': ['Oat']
    }},
    'Histidine_catabolism': {'steps': {
        'Step1_HAL': ['Hal'],
        'Step2_UROC1': ['Uroc1'],
        'Step3_FTCD': ['Ftcd']
    }},
    'Lysine_catabolism': {'steps': {
        'Step1_AASS': ['Aass'],
        'Step2_AASDH': ['Aasdh'],
        'Step3_DHTKD1': ['Dhtkd1']
    }},
    'Phenylalanine_catabolism': {'steps': {
        'Step1_PAH': ['Pah'],
        'Step2_TAT': ['Tat'],
        'Step3_HPD': ['Hpd'],
        'Step4_HGD': ['Hgd']
    }},
    'Tyrosine_catabolism': {'steps': {
        'Step1_TAT': ['Tat'],
        'Step2_HPD': ['Hpd'],
        'Step3_HGD': ['Hgd']
    }},
    'Tryptophan_catabolism': {'steps': {
        'Step1_TDO': ['Tdo2'],
        'Step2_KYNU': ['Kynu'],
        'Step3_HAAO': ['Haao']
    }},
    'Cysteine_catabolism': {'steps': {
        'Step1_CDO1': ['Cdo1'],
        'Step2_GOT': ['Got1', 'Got2']
    }},
    'Asparagine_catabolism': {'steps': {
        'Step1_ASNS': ['Asns'],
        'Step2_GOT': ['Got1', 'Got2']
    }}
}

ALL_PATHWAY_STEPS.update(AA_PATHWAYS)

print("\n" + "="*80)
print("GEOMETRIC MEAN V2 - FULL DATASET ANALYSIS")
print("="*80)
print(f"Samples: {len(sample_cols)}")
print(f"Pathways: {len(ALL_PATHWAY_STEPS)}")

# Step 1: Scale gene expression
print("\nStep 1: Scaling gene expression (Floor: 0.05, Ceiling: 1.0)...")
expr_matrix = data[sample_cols].values
top5_means = np.mean(np.sort(expr_matrix, axis=1)[:, -5:], axis=1)
top5_means[top5_means == 0] = 1.0

scaled_matrix = expr_matrix / top5_means[:, np.newaxis]
scaled_matrix = np.minimum(scaled_matrix, 1.0)
scaled_matrix = np.maximum(scaled_matrix, 0.05)

scaled_data = data.copy()
scaled_data[sample_cols] = scaled_matrix
print(f"✓ Scaled {len(data)} genes")

# Step 2: Calculate pathway flux
print("\nStep 2: Calculating pathway flux (v2: max isozyme, TPI excluded)...")

def calculate_geom_mean_v2(sample, pathways_dict, scaled_data, samples):
    """Calculate geometric mean v2 for all pathways"""
    capacity = {}

    for pathway_name, pathway_info in pathways_dict.items():
        steps = pathway_info['steps']
        step_rates = []

        for step_name, genes in steps.items():
            # Skip TPI for Glycolysis
            if pathway_name == 'Glycolysis' and 'TPI' in step_name.upper():
                continue

            isozyme_exprs = []
            for gene_name in genes:
                gene_row = scaled_data[scaled_data['gene_name'].str.upper() == gene_name.upper()]
                if len(gene_row) > 0 and sample in samples:
                    expr = gene_row[sample].values[0]
                    if not np.isnan(expr):
                        isozyme_exprs.append(expr)

            # v2: MAX of isozymes
            if len(isozyme_exprs) > 0:
                step_rate = np.max(isozyme_exprs)
                step_rates.append(step_rate)

        # Geometric mean
        if len(step_rates) > 0 and all(r > 0 for r in step_rates):
            pathway_flux = np.exp(np.mean(np.log(step_rates)))
            capacity[pathway_name] = pathway_flux
        else:
            capacity[pathway_name] = 0.0

    return capacity

results = []
for i, sample in enumerate(sample_cols):
    if (i + 1) % 50 == 0:
        print(f"  Processing sample {i+1}/{len(sample_cols)}...")

    capacity = calculate_geom_mean_v2(sample, ALL_PATHWAY_STEPS, scaled_data, sample_cols)
    result = {'Sample': sample}
    result.update(capacity)
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_csv('geometric_mean_v2_pathway_fluxes.csv', index=False)
print(f"\n✓ Saved to geometric_mean_v2_pathway_fluxes.csv")
print(f"  Shape: {results_df.shape}")
print(f"  Non-zero samples: {(results_df['Glycolysis'] > 0).sum()}/{len(results_df)}")

# Quick summary
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
for col in ['Glycolysis', 'Oxidative_PPP', 'TCA_Cycle', 'Fatty_Acid_Oxidation', 'Fatty_Acid_Synthesis']:
    if col in results_df.columns:
        values = results_df[col]
        non_zero = values[values > 0]
        print(f"{col:30s}: mean={np.mean(non_zero):.3f}, median={np.median(non_zero):.3f}, n={len(non_zero)}")

print("\n✓ Analysis complete!")
