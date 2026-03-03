#!/usr/bin/env python3
"""
Filter Metabolite_and_Gene_for_cor.csv to keep only genes and metabolites
from central carbon metabolism (TCA cycle, Pentose Phosphate Pathway, and Glycolysis)
based on iMM1415.xml model.
"""

import cobra
import pandas as pd
import re

# Load the metabolic model
print("Loading iMM1415 model...")
model = cobra.io.read_sbml_model('/Users/sunengfu/GSMM/iMM1415.xml')

# Define pathway keywords for identification
tca_keywords = ['citrate synthase', 'aconitate', 'isocitrate', 'ketoglutarate',
                'succinyl', 'succinate', 'fumarate', 'malate', 'oxaloacetate',
                'TCA', 'tricarboxylic', 'Krebs', 'CS', 'ACO', 'IDH', 'AKGD',
                'SUCOAS', 'SUCD', 'FUM', 'MDH']

ppp_keywords = ['glucose-6-phosphate dehydrogenase', 'gluconolactonase',
                '6-phosphogluconate', 'ribulose', 'ribose-5-phosphate',
                'xylulose', 'transketolase', 'transaldolase',
                'pentose phosphate', 'PPP', 'G6PDH', 'PGL', 'GND',
                'RPE', 'RPI', 'TKT', 'TALA']

glycolysis_keywords = ['hexokinase', 'phosphoglucose isomerase',
                       'phosphofructokinase', 'aldolase', 'triose',
                       'glyceraldehyde', 'phosphoglycerate', 'enolase',
                       'pyruvate kinase', 'glucose', 'fructose', 'pyruvate',
                       'glycolysis', 'HK', 'PGI', 'PFK', 'FBP', 'FBA',
                       'TPI', 'GAPD', 'PGK', 'PGM', 'ENO', 'PYK']

def is_pathway_reaction(reaction, keywords):
    """Check if a reaction belongs to a pathway based on keywords."""
    reaction_str = (reaction.id + ' ' + reaction.name).lower()
    return any(keyword.lower() in reaction_str for keyword in keywords)

# Collect reactions from each pathway
print("\nIdentifying pathway reactions...")
tca_reactions = [r for r in model.reactions if is_pathway_reaction(r, tca_keywords)]
ppp_reactions = [r for r in model.reactions if is_pathway_reaction(r, ppp_keywords)]
glycolysis_reactions = [r for r in model.reactions if is_pathway_reaction(r, glycolysis_keywords)]

print(f"Found {len(tca_reactions)} TCA cycle reactions")
print(f"Found {len(ppp_reactions)} PPP reactions")
print(f"Found {len(glycolysis_reactions)} Glycolysis reactions")

# Combine all central carbon metabolism reactions
all_reactions = set(tca_reactions + ppp_reactions + glycolysis_reactions)

# Extract genes and metabolites from these reactions
genes = set()
metabolites = set()

print("\nExtracting genes and metabolites...")
for reaction in all_reactions:
    # Get genes
    for gene in reaction.genes:
        genes.add(gene.name)
        genes.add(gene.id)  # Add both name and ID

    # Get metabolites
    for metabolite in reaction.metabolites:
        # Remove compartment suffixes like _c, _m, _e, etc.
        met_name = metabolite.name
        met_id = re.sub(r'_[a-z]$', '', metabolite.id)
        metabolites.add(met_name)
        metabolites.add(met_id)

print(f"Found {len(genes)} unique genes")
print(f"Found {len(metabolites)} unique metabolites")

# Load the metabolite and gene data
print("\nLoading Metabolite_and_Gene_for_cor.csv...")
df = pd.read_csv('/Users/sunengfu/GSMM/Metabolite_and_Gene_for_cor.csv')

print(f"Original data shape: {df.shape}")
print(f"Rows (metabolites/genes): {len(df)}")
print(f"Columns (samples): {len(df.columns)}")

# Filter rows - keep only genes and metabolites from central carbon metabolism
# The first column contains the names
name_column = df.columns[0]  # Should be '#NAME' or similar

rows_to_keep_mask = []
for name in df[name_column]:
    # Case-insensitive matching
    keep = (name in genes or name in metabolites or
            name.upper() in {g.upper() for g in genes} or
            name.upper() in {m.upper() for m in metabolites} or
            name.lower() in {g.lower() for g in genes} or
            name.lower() in {m.lower() for m in metabolites})
    rows_to_keep_mask.append(keep)

print(f"\nFiltering data...")
print(f"Keeping {sum(rows_to_keep_mask)} out of {len(df)} rows")

# Filter rows
df_filtered = df[rows_to_keep_mask]

print(f"\nFiltered data shape: {df_filtered.shape}")

# Save the filtered data
output_file = '/Users/sunengfu/GSMM/Metabolite_and_Gene_for_cor_central_carbon.csv'
df_filtered.to_csv(output_file, index=False)
print(f"\nFiltered data saved to: {output_file}")

# Print some summary statistics
print("\n=== Summary ===")
print(f"Pathways analyzed: TCA cycle, Pentose Phosphate Pathway, Glycolysis")
print(f"Original data: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Filtered data: {df_filtered.shape[0]} rows x {df_filtered.shape[1]} columns")

# Separate genes and metabolites
kept_names = df_filtered[name_column].tolist()
kept_genes_list = [name for name in kept_names if name in genes or name.upper() in {g.upper() for g in genes}]
kept_metabolites_list = [name for name in kept_names if name in metabolites or name.upper() in {m.upper() for m in metabolites}]

print(f"\nKept {len(kept_genes_list)} genes")
print(f"Kept {len(kept_metabolites_list)} metabolites")

print(f"\nSample of kept genes (first 20):")
print(kept_genes_list[:20] if len(kept_genes_list) > 0 else "No genes found")

print(f"\nSample of kept metabolites (first 20):")
print(kept_metabolites_list[:20] if len(kept_metabolites_list) > 0 else "No metabolites found")
