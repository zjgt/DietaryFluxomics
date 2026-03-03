
import pandas as pd
import numpy as np

def extract_pathway_correlations():
    """
    Extracts individual within-pathway correlation scores from the all-samples correlation matrix.
    """
    pathways = {
        'Glycolysis': {
            'metabolites': [
                'Glucose 6-phosphate', 'Fructose 6-phosphate',
                'D-Glyceraldehyde 3-phosphate', '3-phosphoglycerate', '2-phosphoglycerate',
                'Phosphoenolpyruvate', 'Pyruvate'
            ],
            'genes': [
                'Aldoa', 'Aldob', 'Aldoc', 'Bpgm', 'Eno1', 'Eno2', 'Eno3', 'Gapdhs',
                'Gck', 'Gpi1', 'Hk1', 'Hk2', 'Hk3', 'Hkdc1', 'Pfkl', 'Pfkm', 'Pfkp',
                'Pgam1', 'Pgam2', 'Pgk1', 'Pgk2', 'Pklr', 'Pkm', 'Tpi1'
            ]
        },
        'TCA_Cycle': {
            'metabolites': [
                'Oxalacetate', 'Citrate', 'alpha-Ketoglutarate', 'Succinate', 'Fumarate', 'Malate'
            ],
            'genes': [
                'Aco1', 'Cs', 'Fh1', 'Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g',
                'Mdh1', 'Mdh2', 'Sdha', 'Sdhb'
            ]
        },
        'Pentose_Phosphate_Pathway': {
            'metabolites': [
                'Ribulose 5-phosphate', 'Ribose 5-phosphate',
                'Sedoheptulose 7-phosphate', 'Sedoheptulose 1-phosphate'
            ],
            'genes': [
                'G6pdx', 'H6pd', 'Pgd', 'Pgls', 'Rpe', 'Rpia', 'Taldo1', 'Tkt',
                'Tktl1', 'Tktl2'
            ]
        }
    }

    try:
        corr_matrix = pd.read_csv('correlation_all_samples.csv', index_col=0)
    except FileNotFoundError:
        print("Error: correlation_all_samples.csv not found. Please run the previous analysis first.")
        return

    for pathway_name, members in pathways.items():
        print(f"Processing pathway: {pathway_name}")

        pathway_genes = [g for g in members['genes'] if g in corr_matrix.index]
        pathway_metabolites = [m for m in members['metabolites'] if m in corr_matrix.index]

        gene_gene_scores, metabolite_metabolite_scores, gene_metabolite_scores = [], [], []

        if len(pathway_genes) > 1:
            gene_corr = corr_matrix.loc[pathway_genes, pathway_genes]
            upper_gene = gene_corr.where(np.triu(np.ones(gene_corr.shape), k=1).astype(bool))
            gene_gene_scores = upper_gene.stack().tolist()

        if len(pathway_metabolites) > 1:
            metabolite_corr = corr_matrix.loc[pathway_metabolites, pathway_metabolites]
            upper_metabolite = metabolite_corr.where(np.triu(np.ones(metabolite_corr.shape), k=1).astype(bool))
            metabolite_metabolite_scores = upper_metabolite.stack().tolist()

        if pathway_genes and pathway_metabolites:
            gene_metabolite_corr = corr_matrix.loc[pathway_genes, pathway_metabolites]
            gene_metabolite_scores = gene_metabolite_corr.values.flatten().tolist()

        output_df = pd.DataFrame({
            'gene-gene': pd.Series(gene_gene_scores),
            'gene-metabolite': pd.Series(gene_metabolite_scores),
            'metabolite-metabolite': pd.Series(metabolite_metabolite_scores)
        })

        output_filename = f'{pathway_name}_correlations.csv'
        output_df.to_csv(output_filename, index=False)
        print(f"  Saved scores to {output_filename}")

if __name__ == '__main__':
    extract_pathway_correlations()
