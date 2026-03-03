import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_all_samples_correlation(df, metabolites, genes, results):
    """Performs correlation analysis for all samples combined."""
    print("\nProcessing all samples...")
    features = metabolites + genes
    df_features = df.loc[features]

    # Transpose for correlation calculation
    df_T = df_features.T

    # Full correlation matrix for CSV
    corr_matrix_full = df_T.corr()
    corr_matrix_full.to_csv('correlation_all_samples.csv')
    print("  Saved correlation matrix to correlation_all_samples.csv")

    # Prepare data for heatmap (remove non-variant features)
    non_variant_cols = df_T.columns[df_T.std() == 0]
    if not non_variant_cols.empty:
        print(f"  Found {len(non_variant_cols)} features with no variance, removing them for heatmap generation.")
        df_T_variant = df_T.drop(columns=non_variant_cols)
    else:
        df_T_variant = df_T

    if not df_T_variant.empty:
        corr_matrix_variant = df_T_variant.corr().dropna(how='all').dropna(how='all', axis=1)
        if not corr_matrix_variant.empty:
            try:
                cluster_map = sns.clustermap(
                    corr_matrix_variant,
                    cmap='vlag',
                    vmin=-1,
                    vmax=1,
                    figsize=(16, 16),
                    cbar_pos=(0.92, 0.85, 0.03, 0.15),
                    xticklabels=True,
                    yticklabels=True
                )
                cluster_map.fig.suptitle('Metabolite-Gene Correlation Heatmap for All Samples', y=1.02)
                cluster_map.savefig('correlation_heatmap_all_samples.png')
                plt.close(cluster_map.fig)
                print("  Saved heatmap to correlation_heatmap_all_samples.png")
            except Exception as e:
                print(f"  Could not generate heatmap for all samples. Error: {e}")

            # Calculate average correlation scores
            variant_genes = [g for g in genes if g in corr_matrix_variant.columns]
            variant_metabolites = [m for m in metabolites if m in corr_matrix_variant.columns]

            avg_gene_corr, avg_metabolite_corr, avg_gene_metabolite_corr = np.nan, np.nan, np.nan

            if variant_genes:
                gene_corr = corr_matrix_variant.loc[variant_genes, variant_genes]
                gene_corr_abs = gene_corr.abs()
                upper_gene = gene_corr_abs.where(np.triu(np.ones(gene_corr_abs.shape), k=1).astype(bool))
                avg_gene_corr = upper_gene.stack().mean()

            if variant_metabolites:
                metabolite_corr = corr_matrix_variant.loc[variant_metabolites, variant_metabolites]
                metabolite_corr_abs = metabolite_corr.abs()
                upper_metabolite = metabolite_corr_abs.where(np.triu(np.ones(metabolite_corr_abs.shape), k=1).astype(bool))
                avg_metabolite_corr = upper_metabolite.stack().mean()

            if variant_genes and variant_metabolites:
                gene_metabolite_corr = corr_matrix_variant.loc[variant_genes, variant_metabolites]
                avg_gene_metabolite_corr = gene_metabolite_corr.abs().mean().mean()

            results.append({
                'Tissue': 'All_Samples',
                'Avg_Between_Gene_Correlation': avg_gene_corr,
                'Avg_Between_Metabolite_Correlation': avg_metabolite_corr,
                'Avg_Gene_Metabolite_Correlation': avg_gene_metabolite_corr
            })
        else:
            print("  Correlation matrix is empty after dropping NaN values, skipping heatmap and score calculation.")
    else:
        print("  Skipping heatmap and score calculation as all features have no variance.")

def analyze_tissue_correlations():
    """
    Performs tissue-wide correlation analysis for metabolites and genes.
    """
    try:
        # Load the data
        df = pd.read_csv('Metabolite_and_Gene_for_cor_central_carbon_complete.csv', index_col=0)
    except FileNotFoundError:
        print("Error: Metabolite_and_Gene_for_cor_central_carbon_complete.csv not found.")
        return

    # Get metabolite and gene names based on the specified ranges
    try:
        metabolite_start_index = df.index.get_loc('2-phosphoglycerate')
        metabolite_end_index = df.index.get_loc('UMP')
        gene_start_index = df.index.get_loc('Aco1')
        gene_end_index = df.index.get_loc('Tpi1')

        metabolites = df.index[metabolite_start_index:metabolite_end_index + 1].tolist()
        genes = df.index[gene_start_index:gene_end_index + 1].tolist()
    except KeyError as e:
        print(f"Error: Row name {e} not found in the data. Please check the row names.")
        return

    features = metabolites + genes
    df_features = df.loc[features]

    # Get unique tissue names from column headers
    tissues = sorted(list(set([col.split('_')[0] for col in df.columns])))

    results = []

    for tissue in tissues:
        print(f"Processing tissue: {tissue}")
        
        # Get columns for the current tissue
        tissue_cols = [col for col in df.columns if col.startswith(tissue + '_')]
        
        if not tissue_cols:
            print(f"  No columns found for tissue: {tissue}")
            continue

        # Get data for the tissue
        tissue_df = df_features[tissue_cols]

        # Transpose for correlation calculation (features as columns)
        tissue_df_T = tissue_df.T

        # Calculate the full correlation matrix for CSV output
        corr_matrix_full = tissue_df_T.corr()
        csv_filename = f'correlation_{tissue}.csv'
        corr_matrix_full.to_csv(csv_filename)
        print(f"  Saved correlation matrix to {csv_filename}")

        # Prepare data for the heatmap by removing non-variant features
        non_variant_cols = tissue_df_T.columns[tissue_df_T.std() == 0]
        if not non_variant_cols.empty:
            print(f"  Found {len(non_variant_cols)} features with no variance, removing them for heatmap generation.")
            tissue_df_T_variant = tissue_df_T.drop(columns=non_variant_cols)
        else:
            tissue_df_T_variant = tissue_df_T

        if not tissue_df_T_variant.empty:
            corr_matrix_variant = tissue_df_T_variant.corr()
            # Drop rows/cols from the correlation matrix that are all NaN
            corr_matrix_variant.dropna(axis=0, how='all', inplace=True)
            corr_matrix_variant.dropna(axis=1, how='all', inplace=True)

            if not corr_matrix_variant.empty:
                try:
                    cluster_map = sns.clustermap(
                        corr_matrix_variant,
                        cmap='vlag',
                        vmin=-1,
                        vmax=1,
                        figsize=(16, 16),
                        cbar_pos=(0.92, 0.85, 0.03, 0.15),
                        xticklabels=True,
                        yticklabels=True
                    )
                    cluster_map.fig.suptitle(f'Metabolite-Gene Correlation Heatmap for {tissue}', y=1.02)
                    png_filename = f'correlation_heatmap_{tissue}.png'
                    cluster_map.savefig(png_filename)
                    plt.close(cluster_map.fig)
                    print(f"  Saved heatmap to {png_filename}")
                except Exception as e:
                    print(f"  Could not generate heatmap for {tissue}. Error: {e}")
                
                # Calculate average correlation scores
                variant_genes = [g for g in genes if g in corr_matrix_variant.columns]
                variant_metabolites = [m for m in metabolites if m in corr_matrix_variant.columns]

                avg_gene_corr, avg_metabolite_corr, avg_gene_metabolite_corr = np.nan, np.nan, np.nan

                if variant_genes:
                    gene_corr = corr_matrix_variant.loc[variant_genes, variant_genes]
                    gene_corr_abs = gene_corr.abs()
                    upper_gene = gene_corr_abs.where(np.triu(np.ones(gene_corr_abs.shape), k=1).astype(bool))
                    avg_gene_corr = upper_gene.stack().mean()

                if variant_metabolites:
                    metabolite_corr = corr_matrix_variant.loc[variant_metabolites, variant_metabolites]
                    metabolite_corr_abs = metabolite_corr.abs()
                    upper_metabolite = metabolite_corr_abs.where(np.triu(np.ones(metabolite_corr_abs.shape), k=1).astype(bool))
                    avg_metabolite_corr = upper_metabolite.stack().mean()

                if variant_genes and variant_metabolites:
                    gene_metabolite_corr = corr_matrix_variant.loc[variant_genes, variant_metabolites]
                    avg_gene_metabolite_corr = gene_metabolite_corr.abs().mean().mean()

                results.append({
                    'Tissue': tissue,
                    'Avg_Between_Gene_Correlation': avg_gene_corr,
                    'Avg_Between_Metabolite_Correlation': avg_metabolite_corr,
                    'Avg_Gene_Metabolite_Correlation': avg_gene_metabolite_corr
                })

            else:
                print(f"  Correlation matrix is empty after dropping NaN values, skipping heatmap and score calculation.")
        else:
            print(f"  Skipping heatmap and score calculation for {tissue} as all features have no variance.")

    # Perform analysis for all samples combined
    analyze_all_samples_correlation(df, metabolites, genes, results)

    # Save the average correlation scores
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('average_correlation_scores.csv', index=False)
        print("\nSaved average correlation scores to average_correlation_scores.csv")

if __name__ == '__main__':
    analyze_tissue_correlations()
