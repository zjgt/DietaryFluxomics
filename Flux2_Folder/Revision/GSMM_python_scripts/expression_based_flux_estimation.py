"""
Expression-Based Flux Estimation for Central Carbon Metabolism

Direct approach: estimate relative pathway activity from gene expression
No FBA optimization - just use expression as a proxy for flux capacity

Key assumption: Gene expression correlates with enzyme abundance and flux capacity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("EXPRESSION-BASED FLUX ESTIMATION")
print("=" * 80)

# Load data
data = pd.read_csv('transcriptomics_data.csv', low_memory=False)
with open('data_metadata.json', 'r') as f:
    metadata = json.load(f)

all_sample_cols = metadata['sample_columns']
sample_cols = [col for col in all_sample_cols
               if col not in ['length', 'description', 'gene_name', 'swissprot', 'entrez']]

N_SAMPLES = 10  # Analyze more samples since this is fast
sample_cols = sample_cols[:N_SAMPLES]

print(f"\nAnalyzing {len(sample_cols)} samples")

# Define pathway gene sets by gene name
PATHWAY_GENES = {
    'Glycolysis': {
        'genes': ['Hk1', 'Hk2', 'Hk3', 'Gpi1', 'Pfkl', 'Pfkm', 'Pfkp', 'Aldoa', 'Aldob', 'Aldoc',
                  'Tpi1', 'Gapdh', 'Pgk1', 'Pgam1', 'Pgam2', 'Eno1', 'Eno2', 'Eno3',
                  'Pklr', 'Pkm'],
        'rate_limiting': ['Pfkl', 'Pfkm', 'Pfkp'],  # Phosphofructokinase isoforms
        'description': 'Glucose → Pyruvate'
    },
    'Oxidative_PPP': {
        'genes': ['G6pdx', 'G6pd2', 'Pgls', 'Pgd'],
        'rate_limiting': ['G6pdx', 'G6pd2'],  # G6PD isoforms
        'description': 'NADPH production from glucose-6-phosphate'
    },
    'Non_oxidative_PPP': {
        'genes': ['Rpe', 'Rpia', 'Tkt', 'Tktl1', 'Tktl2', 'Taldo1'],
        'rate_limiting': ['Tkt', 'Taldo1'],
        'description': 'Ribose-5-P interconversion'
    },
    'TCA_Cycle': {
        'genes': ['Cs', 'Aco1', 'Aco2', 'Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g',
                  'Ogdh', 'Dlst', 'Sucla2', 'Suclg1', 'Suclg2', 'Sdha', 'Sdhb', 'Sdhc', 'Sdhd',
                  'Fh1', 'Mdh1', 'Mdh2'],
        'rate_limiting': ['Cs', 'Idh2', 'Ogdh'],  # Citrate synthase, isocitrate DH, α-ketoglutarate DH
        'description': 'Acetyl-CoA → CO2 + NADH'
    },
    'Pyruvate_Dehydrogenase': {
        'genes': ['Pdha1', 'Pdha2', 'Pdhb', 'Dlat', 'Dld', 'Pdk1', 'Pdk2', 'Pdk3', 'Pdk4'],
        'rate_limiting': ['Pdha1', 'Pdhb'],
        'description': 'Pyruvate → Acetyl-CoA'
    },
    'Glutaminolysis': {
        'genes': ['Gls', 'Gls2', 'Glud1', 'Glud2', 'Got1', 'Got2'],
        'rate_limiting': ['Gls', 'Gls2', 'Glud1'],  # Glutaminase, glutamate DH
        'description': 'Glutamine → α-ketoglutarate (TCA entry)'
    },
    'Fatty_Acid_Oxidation': {
        'genes': ['Cpt1a', 'Cpt1b', 'Cpt1c', 'Cpt2', 'Acadvl', 'Acadm', 'Acads',
                  'Hadha', 'Hadhb', 'Echs1', 'Acadl'],
        'rate_limiting': ['Cpt1a', 'Cpt1b', 'Hadha'],  # Carnitine palmitoyltransferase I
        'description': 'Fatty acids → Acetyl-CoA'
    },
    'Fatty_Acid_Synthesis': {
        'genes': ['Acaca', 'Acacb', 'Fasn', 'Scd1', 'Scd2', 'Elovl5', 'Elovl6'],
        'rate_limiting': ['Acaca', 'Fasn', 'Scd1'],  # ACC, FAS, SCD
        'description': 'Acetyl-CoA → Palmitate/Oleate'
    },
    'Malic_Enzyme': {
        'genes': ['Me1', 'Me2', 'Me3'],
        'rate_limiting': ['Me1', 'Me2'],
        'description': 'Alternative NADPH source (Malate → Pyruvate)'
    },
    'NADPH_Isocitrate_DH': {
        'genes': ['Idh1', 'Idh2'],
        'rate_limiting': ['Idh1', 'Idh2'],
        'description': 'Alternative NADPH source (in TCA)'
    }
}

# Calculate pathway activities
print("\n" + "=" * 80)
print("CALCULATING PATHWAY ACTIVITIES FROM GENE EXPRESSION")
print("=" * 80)

results = []

for sample in sample_cols:
    print(f"\n{sample}:")

    pathway_activities = {'sample': sample}

    for pathway_name, pathway_info in PATHWAY_GENES.items():
        # Find genes in data
        gene_expr = []
        rate_lim_expr = []

        for gene_name in pathway_info['genes']:
            gene_row = data[data['gene_name'].str.upper() == gene_name.upper()]
            if len(gene_row) > 0:
                expr = gene_row[sample].values[0]
                gene_expr.append(expr)

                # Check if rate-limiting
                if gene_name in pathway_info['rate_limiting']:
                    rate_lim_expr.append(expr)

        # Calculate metrics
        if len(gene_expr) > 0:
            mean_expr = np.mean(gene_expr)
            median_expr = np.median(gene_expr)
            max_expr = np.max(gene_expr)

            # Rate-limiting step expression (bottleneck)
            if len(rate_lim_expr) > 0:
                rate_limit_expr = np.mean(rate_lim_expr)
            else:
                rate_limit_expr = mean_expr

            # Pathway capacity estimate: use rate-limiting enzymes
            # This is the "flux capacity" of the pathway
            pathway_capacity = rate_limit_expr

        else:
            mean_expr = 0
            median_expr = 0
            max_expr = 0
            rate_limit_expr = 0
            pathway_capacity = 0

        pathway_activities[f'{pathway_name}_mean'] = mean_expr
        pathway_activities[f'{pathway_name}_median'] = median_expr
        pathway_activities[f'{pathway_name}_capacity'] = pathway_capacity
        pathway_activities[f'{pathway_name}_genes_found'] = len(gene_expr)

        print(f"  {pathway_name:25s}: capacity={pathway_capacity:8.2f} TPM (mean={mean_expr:7.2f}, {len(gene_expr):2d} genes)")

    results.append(pathway_activities)

# Create results dataframe
results_df = pd.DataFrame(results)

# Save detailed results
results_df.to_csv('expression_based_pathway_activities.csv', index=False)
print("\nSaved: expression_based_pathway_activities.csv")

# Calculate key ratios
results_df['oxPPP_Glyc_ratio'] = results_df['Oxidative_PPP_capacity'] / results_df['Glycolysis_capacity']
results_df['TCA_Glyc_ratio'] = results_df['TCA_Cycle_capacity'] / results_df['Glycolysis_capacity']
results_df['FAO_FAS_ratio'] = results_df['Fatty_Acid_Oxidation_capacity'] / results_df['Fatty_Acid_Synthesis_capacity'].replace(0, np.nan)

# Calculate NADPH production capacity
results_df['NADPH_production'] = (results_df['Oxidative_PPP_capacity'] +  # oxPPP contribution
                                   results_df['Malic_Enzyme_capacity'] +      # ME contribution
                                   results_df['NADPH_Isocitrate_DH_capacity']) # IDH contribution

print("\n" + "=" * 80)
print("PATHWAY CAPACITY SUMMARY (Rate-Limiting Enzyme Expression)")
print("=" * 80)

summary_cols = ['sample', 'Glycolysis_capacity', 'Oxidative_PPP_capacity',
                'TCA_Cycle_capacity', 'Pyruvate_Dehydrogenase_capacity',
                'Glutaminolysis_capacity', 'Fatty_Acid_Oxidation_capacity']
print(results_df[summary_cols].to_string(index=False))

print("\n" + "=" * 80)
print("METABOLIC RATIOS")
print("=" * 80)

ratio_cols = ['sample', 'oxPPP_Glyc_ratio', 'TCA_Glyc_ratio', 'FAO_FAS_ratio']
print(results_df[ratio_cols].to_string(index=False))

print("\n" + "=" * 80)
print("NADPH PRODUCTION SOURCES")
print("=" * 80)

nadph_cols = ['sample', 'Oxidative_PPP_capacity', 'Malic_Enzyme_capacity',
              'NADPH_Isocitrate_DH_capacity', 'NADPH_production']
print(results_df[nadph_cols].to_string(index=False))

# Comprehensive visualization
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# 1. Central carbon pathways
ax1 = fig.add_subplot(gs[0, :2])
pathway_cols = ['Glycolysis_capacity', 'Oxidative_PPP_capacity',
                'Non_oxidative_PPP_capacity', 'TCA_Cycle_capacity']
results_df[pathway_cols].plot(kind='bar', ax=ax1, width=0.8)
ax1.set_xticklabels(results_df['sample'], rotation=45, ha='right')
ax1.set_ylabel('Expression (TPM)', fontsize=11)
ax1.set_title('Central Carbon Metabolism - Pathway Capacity', fontsize=12, fontweight='bold')
ax1.legend(['Glycolysis', 'Oxidative PPP', 'Non-ox PPP', 'TCA'], fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# 2. Substrate utilization
ax2 = fig.add_subplot(gs[0, 2])
substrate_cols = ['Pyruvate_Dehydrogenase_capacity', 'Glutaminolysis_capacity',
                  'Fatty_Acid_Oxidation_capacity']
results_df[substrate_cols].plot(kind='bar', ax=ax2, width=0.8, stacked=True)
ax2.set_xticklabels(results_df['sample'], rotation=45, ha='right')
ax2.set_ylabel('Expression (TPM)', fontsize=11)
ax2.set_title('Substrate Oxidation Capacity', fontsize=12, fontweight='bold')
ax2.legend(['PDH (Pyruvate)', 'Glutaminolysis', 'β-oxidation'], fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# 3. Glycolysis vs oxPPP
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(results_df))
width = 0.35
ax3.bar(x - width/2, results_df['Glycolysis_capacity'], width, label='Glycolysis', color='steelblue')
ax3.bar(x + width/2, results_df['Oxidative_PPP_capacity'], width, label='oxPPP', color='coral')
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['sample'], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Expression (TPM)', fontsize=11)
ax3.set_title('Glycolysis vs Oxidative PPP', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# 4. oxPPP/Glycolysis ratio
ax4 = fig.add_subplot(gs[1, 1])
colors = ['green' if r > 0.1 else 'orange' if r > 0.05 else 'lightcoral'
          for r in results_df['oxPPP_Glyc_ratio']]
ax4.bar(range(len(results_df)), results_df['oxPPP_Glyc_ratio'], color=colors)
ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, label='Typical ratio (~0.1)')
ax4.axhline(y=0.05, color='orange', linestyle=':', linewidth=1.5, label='Low ratio (0.05)')
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(results_df['sample'], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Ratio', fontsize=11)
ax4.set_title('oxPPP/Glycolysis Expression Ratio', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# 5. TCA/Glycolysis ratio
ax5 = fig.add_subplot(gs[1, 2])
ax5.bar(range(len(results_df)), results_df['TCA_Glyc_ratio'], color='purple')
ax5.set_xticks(range(len(results_df)))
ax5.set_xticklabels(results_df['sample'], rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('Ratio', fontsize=11)
ax5.set_title('TCA/Glycolysis Expression Ratio', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. NADPH production sources
ax6 = fig.add_subplot(gs[2, :])
nadph_sources = ['Oxidative_PPP_capacity', 'Malic_Enzyme_capacity', 'NADPH_Isocitrate_DH_capacity']
results_df[nadph_sources].plot(kind='bar', stacked=True, ax=ax6, width=0.8)
ax6.set_xticklabels(results_df['sample'], rotation=45, ha='right')
ax6.set_ylabel('Expression (TPM)', fontsize=11)
ax6.set_title('NADPH Production Capacity by Source', fontsize=12, fontweight='bold')
ax6.legend(['Oxidative PPP (G6PD)', 'Malic Enzyme', 'Isocitrate DH'], fontsize=9)
ax6.grid(axis='y', alpha=0.3)

# 7. FAO vs FAS
ax7 = fig.add_subplot(gs[3, 0])
x = np.arange(len(results_df))
width = 0.35
ax7.bar(x - width/2, results_df['Fatty_Acid_Oxidation_capacity'], width,
        label='FA Oxidation', color='indianred')
ax7.bar(x + width/2, results_df['Fatty_Acid_Synthesis_capacity'], width,
        label='FA Synthesis', color='lightgreen')
ax7.set_xticks(x)
ax7.set_xticklabels(results_df['sample'], rotation=45, ha='right', fontsize=8)
ax7.set_ylabel('Expression (TPM)', fontsize=11)
ax7.set_title('Lipid Metabolism: Oxidation vs Synthesis', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(axis='y', alpha=0.3)

# 8. Sample clustering heatmap
ax8 = fig.add_subplot(gs[3, 1:])
heatmap_cols = ['Glycolysis_capacity', 'Oxidative_PPP_capacity', 'TCA_Cycle_capacity',
                'Pyruvate_Dehydrogenase_capacity', 'Glutaminolysis_capacity',
                'Fatty_Acid_Oxidation_capacity', 'Fatty_Acid_Synthesis_capacity']
heatmap_data = results_df[heatmap_cols].T

# Normalize by row for better visualization
heatmap_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)

sns.heatmap(heatmap_norm, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Relative Expression'},
            xticklabels=results_df['sample'], yticklabels=[c.replace('_capacity', '') for c in heatmap_cols],
            ax=ax8)
ax8.set_title('Pathway Expression Heatmap (Normalized)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Sample', fontsize=11)

plt.suptitle('Expression-Based Pathway Flux Capacity Estimation',
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('expression_based_pathway_activities.png', dpi=300, bbox_inches='tight')
print("\nSaved: expression_based_pathway_activities.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print(f"\nKey Findings:")
print(f"  - Mean oxPPP/Glycolysis ratio: {results_df['oxPPP_Glyc_ratio'].mean():.3f}")
print(f"  - Samples with oxPPP/Glyc > 0.1: {(results_df['oxPPP_Glyc_ratio'] > 0.1).sum()}/{len(results_df)}")
print(f"  - Mean TCA/Glycolysis ratio: {results_df['TCA_Glyc_ratio'].mean():.2f}")
print(f"\nTo analyze all {len(all_sample_cols)-5} samples, set N_SAMPLES = {len(all_sample_cols)-5}")
