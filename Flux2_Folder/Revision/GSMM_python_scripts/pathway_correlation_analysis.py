
import pandas as pd
import numpy as np

def analyze_pathway_correlations():
    """
    Performs pathway-specific correlation analysis for each tissue.
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
        'TCA Cycle': {
            'metabolites': [
                'Oxalacetate', 'Citrate', 'alpha-Ketoglutarate', 'Succinate', 'Fumarate', 'Malate'
            ],
            'genes': [
                'Aco1', 'Cs', 'Fh1', 'Idh1', 'Idh2', 'Idh3a', 'Idh3b', 'Idh3g',
                'Mdh1', 'Mdh2', 'Sdha', 'Sdhb'
            ]
        },
        'Pentose Phosphate Pathway': {
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
        df = pd.read_csv('Metabolite_and_Gene_for_cor_central_carbon_complete.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Metabolite_and_Gene_for_cor_central_carbon_complete.csv not found.")
        return

    tissues = sorted(list(set([col.split('_')[0] for col in df.columns])))
    all_results = []

    for tissue in tissues:
        print(f"Processing tissue: {tissue}")
        tissue_cols = [col for col in df.columns if col.startswith(tissue + '_')]
        if not tissue_cols:
            continue

        tissue_df = df[tissue_cols]
        tissue_df_T = tissue_df.T
        
        non_variant_cols = tissue_df_T.columns[tissue_df_T.std() == 0]
        tissue_df_T_variant = tissue_df_T.drop(columns=non_variant_cols, errors='ignore')

        if tissue_df_T_variant.empty:
            continue

        corr_matrix = tissue_df_T_variant.corr()

        for pathway_name, members in pathways.items():
            pathway_genes = [g for g in members['genes'] if g in corr_matrix.columns]
            pathway_metabolites = [m for m in members['metabolites'] if m in corr_matrix.columns]

            avg_gene_corr, avg_metabolite_corr, avg_gene_metabolite_corr = np.nan, np.nan, np.nan

            if len(pathway_genes) > 1:
                gene_corr = corr_matrix.loc[pathway_genes, pathway_genes]
                #gene_corr_abs = gene_corr.abs()
                upper_gene = gene_corr.where(np.triu(np.ones(gene_corr.shape), k=1).astype(bool))
                avg_gene_corr = upper_gene.stack().mean()

            if len(pathway_metabolites) > 1:
                metabolite_corr = corr_matrix.loc[pathway_metabolites, pathway_metabolites]
                #metabolite_corr_abs = metabolite_corr.abs()
                upper_metabolite = metabolite_corr.where(np.triu(np.ones(metabolite_corr.shape), k=1).astype(bool))
                avg_metabolite_corr = upper_metabolite.stack().mean()

            if pathway_genes and pathway_metabolites:
                gene_metabolite_corr = corr_matrix.loc[pathway_genes, pathway_metabolites]
                avg_gene_metabolite_corr = gene_metabolite_corr.abs().mean().mean()

            all_results.append({
                'Tissue': tissue,
                'Pathway': pathway_name,
                'Avg_Between_Gene_Correlation': avg_gene_corr,
                'Avg_Between_Metabolite_Correlation': avg_metabolite_corr,
                'Avg_Gene_Metabolite_Correlation': avg_gene_metabolite_corr
            })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('pathway_average_correlation_scores.csv', index=False)
        print("\nSaved pathway-specific average correlation scores to pathway_average_correlation_scores.csv")

if __name__ == '__main__':
    analyze_pathway_correlations()
