#!/usr/bin/env python3
"""
Filter to keep genes from core central carbon metabolism (from model)
PLUS complete list of metabolite intermediates from glycolysis, TCA, and PPP.
"""

import cobra
import pandas as pd
import re

# Load the metabolic model to get genes
print("Loading iMM1415 model...")
model = cobra.io.read_sbml_model('/Users/sunengfu/GSMM/iMM1415.xml')

# Core enzyme patterns for finding reactions (same as before)
core_enzyme_patterns = [
    # Glycolysis
    r'\bHK\b', r'\bhexokinase\b', r'\bglckinase\b',
    r'\bPGI\b', r'\bphosphoglucose.?isomerase\b',
    r'\bPFK\b', r'\bphosphofructokinase\b',
    r'\bFBA\b', r'\baldolase\b', r'\bfructose.?bisphosphate.?aldolase\b',
    r'\bTPI\b', r'\btriose.?phosphate.?isomerase\b',
    r'\bGAPDH\b', r'\bGAPD\b', r'\bglyceraldehyde.?3.?phosphate.?dehydrogenase\b',
    r'\bPGK\b', r'\bphosphoglycerate.?kinase\b',
    r'\bPGM\b', r'\bphosphoglycerate.?mutase\b',
    r'\bENO\b', r'\benolase\b',
    r'\bPYK\b', r'\bpyruvate.?kinase\b', r'\bPK\b',
    # TCA
    r'\bCS\b', r'\bcitrate.?synthase\b', r'\bcitrate.?oxaloacetate\b',
    r'\bACO\b', r'\baconitase\b', r'\baconitate\b',
    r'\bIDH\b', r'\bisocitrate.?dehydrogenase\b',
    r'\bAKGD\b', r'\bketoglutarate.?dehydrogenase\b', r'\bOGDH\b',
    r'\bSUCOAS\b', r'\bsuccinyl.?coa.?synthetase\b', r'\bSUCLA\b', r'\bSUCLG\b',
    r'\bSDH\b', r'\bsuccinate.?dehydrogenase\b', r'\bSUCD\b',
    r'\bFUM\b', r'\bfumarase\b', r'\bfumarate.?hydratase\b',
    r'\bMDH\b', r'\bmalate.?dehydrogenase\b',
    # PPP
    r'\bG6PDH\b', r'\bglucose.?6.?phosphate.?dehydrogenase\b',
    r'\bPGL\b', r'\bgluconolactonase\b', r'\b6.?phosphogluconolactonase\b',
    r'\bGND\b', r'\bphosphogluconate.?dehydrogenase\b', r'\b6.?phosphogluconate.?dehydrogenase\b',
    r'\bRPE\b', r'\bribulose.?5.?phosphate.?epimerase\b',
    r'\bRPI\b', r'\bribose.?5.?phosphate.?isomerase\b',
    r'\bTKT\b', r'\btransketolase\b',
    r'\bTALA\b', r'\btransaldolase\b'
]

# Define core metabolites for matching in reactions
core_metabolite_patterns = [
    'glucose', 'glc', 'g6p', 'glucose-6-phosphate',
    'f6p', 'fructose-6-phosphate', 'fdp', 'f16bp',
    'dhap', 'dihydroxyacetone', 'g3p', 'gap', 'glyceraldehyde',
    '13dpg', '3pg', '2pg', 'pep', 'phosphoenolpyruvate',
    'pyr', 'pyruvate', 'acetyl', 'cit', 'citrate',
    'isocitrate', 'icit', 'ketoglutarate', 'akg',
    'succinate', 'succ', 'fumarate', 'fum', 'malate', 'mal',
    'oxaloacetate', 'oaa', '6pg', 'ribulose', 'ribose',
    'xylulose', 'sedoheptulose', 'erythrose'
]

print("\n=== Step 1: Finding core central carbon metabolism reactions ===")

# Find reactions that match core pathways
core_reactions = []

for reaction in model.reactions:
    rxn_string = (reaction.id + ' ' + reaction.name).lower()
    rxn_metabolites = {m.name.lower() for m in reaction.metabolites}
    rxn_metabolite_ids = {re.sub(r'_[a-z]$', '', m.id).lower() for m in reaction.metabolites}

    # Check metabolite match
    metabolite_match = False
    for core_met in core_metabolite_patterns:
        if any(core_met.lower() in met for met in rxn_metabolites) or \
           any(core_met.lower() in met for met in rxn_metabolite_ids):
            metabolite_match = True
            break

    # Check enzyme pattern match
    enzyme_match = any(re.search(pattern, rxn_string, re.IGNORECASE)
                      for pattern in core_enzyme_patterns)

    if metabolite_match and enzyme_match:
        core_reactions.append(reaction)

print(f"Found {len(core_reactions)} core central carbon metabolism reactions")

# Extract genes from these reactions
print("\n=== Step 2: Extracting genes from core reactions ===")

core_genes = set()
for reaction in core_reactions:
    for gene in reaction.genes:
        if gene.name and gene.name != '':
            core_genes.add(gene.name)
        core_genes.add(gene.id)

print(f"Extracted {len(core_genes)} genes")
print("Sample genes:", sorted(core_genes)[:20])

# Now define COMPREHENSIVE metabolite list with all synonyms
print("\n=== Step 3: Defining comprehensive metabolite list ===")

# Complete list of central carbon metabolism metabolites with all common names/synonyms
comprehensive_metabolites = {
    # GLYCOLYSIS
    'Glucose', 'D-Glucose', 'Glc',
    'Glucose-6-phosphate', 'Glucose 6-phosphate', 'G6P', 'D-Glucose 6-phosphate',
    'Fructose-6-phosphate', 'Fructose 6-phosphate', 'F6P', 'D-Fructose 6-phosphate',
    'Fructose-1,6-bisphosphate', 'Fructose 1,6-bisphosphate', 'F1,6BP', 'FBP', 'F16BP',
    'D-Fructose 1,6-bisphosphate',
    'Dihydroxyacetone phosphate', 'DHAP', 'Glycerone phosphate', 'glycerone-P',
    'Glyceraldehyde-3-phosphate', 'Glyceraldehyde 3-phosphate', 'G3P', 'GAP', 'GADP',
    'D-Glyceraldehyde 3-phosphate', 'D-Glyceraldehyde-3-phosphate',
    '1,3-Bisphosphoglycerate', '1,3BPG', '1,3-BPG', '13dpg',
    '3-Phosphoglycerate', '3-Phospho-D-glycerate', '3PG', '3pg', 'glycerate-3P',
    '3-Phospho-D-glyceroyl phosphate',
    '2-Phosphoglycerate', '2-Phospho-D-glycerate', '2PG', '2pg',
    'D-Glycerate 2-phosphate',
    'Phosphoenolpyruvate', 'PEP',
    'Pyruvate', 'Pyr',

    # TCA CYCLE
    'Acetyl-CoA', 'Acetyl CoA', 'AcCoA', 'Acetylcoenzyme A',
    'Oxaloacetate', 'Oxalacetate', 'OAA',
    'Citrate', 'Cit',
    'cis-Aconitate', 'Aconitate',
    'Isocitrate', 'Icit',
    'α-Ketoglutarate', 'alpha-Ketoglutarate', 'a-Ketoglutarate',
    '2-Oxoglutarate', '2-oxoglutarate', 'AKG', 'aKG',
    'Succinyl-CoA', 'Succinyl CoA', 'SucCoA',
    'Succinate', 'Succ',
    'Fumarate', 'Fum',
    'Malate', 'L-Malate', 'Mal',

    # PENTOSE PHOSPHATE PATHWAY
    '6-Phosphogluconate', '6-Phospho-D-gluconate', '6PG', '6pgc',
    'D-6-Phosphogluconate',
    '6-Phosphogluconolactone', '6-phospho-D-glucono-1,5-lactone', '6PGL',
    'Ribulose-5-phosphate', 'Ribulose 5-phosphate', 'Ru5P', 'D-Ribulose 5-phosphate',
    'D-Ribulose-5-phosphate',
    'Ribose-5-phosphate', 'Ribose 5-phosphate', 'R5P', 'D-Ribose 5-phosphate',
    'Alpha-D-Ribose 5-phosphate', 'D-Ribose-5-phosphate',
    'Xylulose-5-phosphate', 'Xylulose 5-phosphate', 'X5P', 'Xu5P',
    'D-Xylulose 5-phosphate', 'D-Xylulose-5-phosphate',
    'Sedoheptulose-7-phosphate', 'Sedoheptulose 7-phosphate', 'S7P',
    'Sedoheptulose 1-phosphate',
    'Erythrose-4-phosphate', 'Erythrose 4-phosphate', 'E4P',
    'D-Erythrose 4-phosphate', 'D-Erythrose-4-phosphate',

    # COFACTORS & ENERGY
    'ATP', 'ADP', 'AMP',
    'NAD', 'NAD+', 'NADH',
    'NADP', 'NADP+', 'NADPH',
    'FAD', 'FADH2',
    'CoA', 'Coenzyme A', 'CoASH',
    'Pi', 'Phosphate', 'Orthophosphate',
    'H2O', 'Water',
    'CO2', 'Carbon dioxide',

    # Related compounds
    'GDP-L-fucose',
    'Fructose',
    'D-Fructose',
    'IMP',
    'GMP',
    'CMP',
    'UMP',
    'dTMP'
}

print(f"Defined {len(comprehensive_metabolites)} metabolite names/synonyms")

# Now filter the CSV files
print("\n=== Step 4: Loading and filtering CSV files ===")

# Load dfT_corr.csv
df_corr = pd.read_csv('/Users/sunengfu/GSMM/dfT_corr.csv', index_col=0)
print(f"\ndfT_corr.csv original shape: {df_corr.shape}")

# Filter columns - keep if it's a core gene OR a core metabolite
def is_core_component(name, core_genes, core_metabolites):
    """Check if name matches a core gene or metabolite (case-insensitive)"""
    # Check genes
    if name in core_genes:
        return True
    if name.upper() in {g.upper() for g in core_genes if g}:
        return True

    # Check metabolites (case-insensitive)
    name_upper = name.upper()
    for met in core_metabolites:
        if name == met or name_upper == met.upper():
            return True

    return False

columns_to_keep = [col for col in df_corr.columns
                   if is_core_component(col, core_genes, comprehensive_metabolites)]
rows_to_keep = [idx for idx in df_corr.index
                if is_core_component(idx, core_genes, comprehensive_metabolites)]

print(f"Keeping {len(columns_to_keep)} columns")
print(f"Keeping {len(rows_to_keep)} rows")

df_corr_filtered = df_corr.loc[rows_to_keep, columns_to_keep]
print(f"Filtered shape: {df_corr_filtered.shape}")

output_file1 = '/Users/sunengfu/GSMM/dfT_corr_central_carbon_complete.csv'
df_corr_filtered.to_csv(output_file1)
print(f"Saved to: {output_file1}")

# Filter Metabolite_and_Gene_for_cor.csv
print("\n=== Step 5: Filtering Metabolite_and_Gene_for_cor.csv ===")

df_met_gene = pd.read_csv('/Users/sunengfu/GSMM/Metabolite_and_Gene_for_cor.csv')
print(f"Original shape: {df_met_gene.shape}")

name_column = df_met_gene.columns[0]
rows_to_keep_mask = [is_core_component(name, core_genes, comprehensive_metabolites)
                     for name in df_met_gene[name_column]]

df_met_gene_filtered = df_met_gene[rows_to_keep_mask]
print(f"Filtered shape: {df_met_gene_filtered.shape}")

output_file2 = '/Users/sunengfu/GSMM/Metabolite_and_Gene_for_cor_central_carbon_complete.csv'
df_met_gene_filtered.to_csv(output_file2, index=False)
print(f"Saved to: {output_file2}")

# Final summary
print("\n" + "="*70)
print("=== FINAL SUMMARY ===")
print("="*70)
print(f"Core reactions identified: {len(core_reactions)}")
print(f"Core genes identified: {len(core_genes)}")
print(f"Comprehensive metabolite names defined: {len(comprehensive_metabolites)}")
print(f"\ndfT_corr.csv: {df_corr.shape} -> {df_corr_filtered.shape}")
print(f"Metabolite_and_Gene_for_cor.csv: {df_met_gene.shape} -> {df_met_gene_filtered.shape}")

# Show what was kept
kept_names = df_met_gene_filtered[name_column].tolist()
print(f"\n=== Items kept in filtered files ({len(kept_names)} total) ===")

# Separate genes and metabolites
kept_genes_list = [name for name in kept_names if is_core_component(name, core_genes, set())]
kept_metabolites_list = [name for name in kept_names if name not in kept_genes_list]

print(f"\nGenes ({len(kept_genes_list)}):")
print(sorted(kept_genes_list))

print(f"\nMetabolites ({len(kept_metabolites_list)}):")
print(sorted(kept_metabolites_list))
