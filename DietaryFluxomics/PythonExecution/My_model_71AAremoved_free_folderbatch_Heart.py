#-------------------------------------------------------------------------------
# Name:        mfapy example 3 13C-MFA of brest cancer (MCF-7) cells
#              Metabolic model and data used in this code is derived from Araki et al Mass Spectrometry 2018, 7, A0067.
#
# Author:      Fumio_Matsuda
#
# Created:     12/06/2018
# Copyright:   (c) Fumio_Matsuda 2018
# Licence:     MIT license
#-------------------------------------------------------------------------------
import os, sys, time
import mfapy
import pandas as pd

start = time.time()

if __name__ == '__main__':
    #
    # Construction of metabolic model
    start = time.time()
    print(start)
    #
    reactions, reversible, metabolites, target_fragments = mfapy.mfapyio.load_metabolic_model("My_model_v1AAremoved_v2.txt")
    model = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, target_fragments)
    #
    # Configurations
    #
    model.set_configuration(callbacklevel = 4)
    model.set_configuration(iteration_max = 1000000) # Maximal iternations in optimization
    model.set_configuration(number_of_repeat = 3) #Iteration in self.fitting_flux(method = 'deep') [SLSQP => LN_PRAXIS] * n
    model.set_configuration(ncpus = 5) #Number of local CPUs for Parallel python
    #
    # Loading metabolite state from text file to constrain flux vector
    #
    state_dic = model.load_states("My_status_v71AAremoved_Heart.csv", format = 'csv')
    model.set_constraints_from_state_dict(state_dic)
    model.update()
    carbon_ratio = pd.read_csv('Tissue_Carbon_Source_v2.csv', index_col=0)

    source_dir1 = 'D:/mfapyt/sample/MDV_Tissue/Heart/'
    dest_dir1 = 'D:/mfapyt/sample/Results_Tissue/Heart/'
    files1 = os.listdir(source_dir1)

    for file in files1:
        print(file)
        result_path = dest_dir1 + file
        C12 = carbon_ratio.loc['C12', file]
        C13 = carbon_ratio.loc['C13', file]
        carbon_source1 = model.generate_carbon_source_template()
        carbon_source1.set_all_isotopomers('SubsCO2', [0.99, 0.01])
        carbon_source1.set_each_isotopomer('SubsGlc', {'#000000': C12, '#111111': C13}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsGln', {'#00000': 1.}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsSer', {'#000': 1.}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsAla', {'#000': 1.}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsLac', {'#000': C12, '#111': C13}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsR5P', {'#00000': 1.}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsAcCOAmit', {'#00': 1.}, correction='yes')
        carbon_source1.set_each_isotopomer('SubsGly', {'#000': 1.}, correction='yes')
        mdv_observed1 = model.load_mdv_data(source_dir1 + file)
        mdv_observed1.set_std(0.015, method='absolute')
        model.set_experiment('ex1', mdv_observed1, carbon_source1)  #
        method_global = "GN_CRS2_LM"
        #methods = ["SLSQP", "COBYLA", "LN_COBYLA", "LN_BOBYQA", "LN_NEWUOA", "LN_PRAXIS", "LN_SBPLX",
        #           "LN_NELDERMEAD", "GN_CRS2_LM", "deep"]
        method_local = "SLSQP"
        state1, flux_opt1 = model.generate_initial_states(1000, 5, method="parallel")
        results = [('template', flux_opt1[0])]
        
        state, RSS_bestfit, flux_opt1 = model.fitting_flux(method=method_global, flux=flux_opt1)
        for i in range(5):
            results.append((method_global, flux_opt1[i]))

        state, RSS_bestfit, flux_opt_slsqp = model.fitting_flux(method=method_local, flux=flux_opt1)
        for i in range(5):
            pvalue, rss_thres = model.goodness_of_fit(flux_opt_slsqp[i], alpha=0.05)
            results.append((method_local, flux_opt_slsqp[i]))


        model.show_results(results, filename=result_path + '.csv', format="csv")
        model.show_results(results, pool_size="off", checkrss="on")


end = time.time()
duration = end - start
print(duration)
