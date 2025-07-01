#-------------------------------------------------------------------------------
# Name:        Flux2_grid_search_mouse.py
#
# Author:      Fumio_Matsuda, Suneng_Fu
#
# Created:     12/06/2018, 06/30/2025
# Copyright:   (c) Fumio_Matsuda 2018, Suneng_Fu 2025
# Licence:     MIT license
#-------------------------------------------------------------------------------
import os, sys, time
import mfapy
import pandas as pd
import numpy as np

start = time.localtime()

if __name__ == '__main__':
    #
    # Construction of metabolic model
    start = time.localtime()
    print(start)
    #
    reactions, reversible, metabolites, target_fragments = mfapy.mfapyio.load_metabolic_model("Flux2_model_mouse.txt")
    model = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, target_fragments)
    #
    # Configurations
    #
    model.set_configuration(callbacklevel = 4)
    model.set_configuration(iteration_max = 1000000) # Maximal iterations in optimization
    model.set_configuration(number_of_repeat = 3) #Iteration in self.fitting_flux(method = 'deep') [SLSQP => LN_PRAXIS] * n
    model.set_configuration(ncpus = 20) #Number of local CPUs for Parallel python
    #
    # Loading metabolite state from text file to constrain flux vector
    #
    state_dic = model.load_states("Flux2_status_mouse.csv", format = 'csv')
    model.set_constraints_from_state_dict(state_dic)
    model.update()
    carbon_ratio = pd.read_csv('Flux2_Carbon_Source_Mouse.csv', index_col=0)

    source_dir1 = 'Flux2_MDV_Mouse_Files/'
    dest_dir1 = 'Flux2_Results/'
    files1 = os.listdir(source_dir1)
    files1 = np.sort(files1)


    for file in files1:
        if file.endswith('.DS_Store'):
            continue
        else:
            print(file)
            result_path = dest_dir1 + file
            C12G = carbon_ratio.loc['C12_G', file] - 0.1
            C13G = 1 - C12G
            C12L = carbon_ratio.loc['C12_L', file] - 0.1
            C13L = 1 - C12L
            C12_G = C12G + 0.01 * Glc
            C13_G = 1 - C12_G
            C12_L = C12L + 0.01 * Lac
            C13_L = 1 - C12_L
            carbon_source1 = model.generate_carbon_source_template()
            # carbon_source1.set_all_isotopomers('SubsCO2', [0.99, 0.01])
            carbon_source1.set_each_isotopomer('SubsGlc', {'#000000': C12_G, '#111111': C13_G},
                                               correction='yes')
            carbon_source1.set_each_isotopomer('SubsGln', {'#00000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsBCAA', {'#00000': 1.}, correction='yes')
            # carbon_source1.set_each_isotopomer('SubsSer', {'#000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsAla', {'#000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsLac', {'#000': C12_L, '#111': C13_L}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsR5P', {'#00000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsAcCOAmit', {'#00': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsAcAsp', {'#000000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsGly', {'#000': 1.}, correction='yes')
            carbon_source1.set_each_isotopomer('SubsGlycogen', {'#000000': 1.}, correction='yes')
            mdv_observed1 = model.load_mdv_data(source_dir1 + file)
            mdv_observed1.set_std(0.015, method='absolute')
            model.set_experiment('ex1', mdv_observed1, carbon_source1)  #
            method_global = "GN_CRS2_LM"
            # method_global = "GN_ISRES"
            # method_global = "deep"
            # methods = ["SLSQP", "COBYLA", "LN_COBYLA", "LN_BOBYQA", "LN_NEWUOA", "LN_PRAXIS", "LN_SBPLX",
            #           "LN_NELDERMEAD", "GN_CRS2_LM", "deep"]
            method_local = "SLSQP"
            state1, flux_opt1 = model.generate_initial_states(2000, 20, method="parallel")
            # res_out = flux_opt1[0]['reaction']['r001_SubsGlc']['value']
            # print(res_out)
            size = len(flux_opt1)
            try:
                results = [('template', flux_opt1[0])]
                state, RSS_bestfit, flux_opt1 = model.fitting_flux(method=method_global, flux=flux_opt1)
                size_global = len(flux_opt1)
                for i in range(size_global):
                    results.append((method_global, flux_opt1[i]))
                state, RSS_bestfit, flux_opt_slsqp = model.fitting_flux(method=method_local, flux=flux_opt1)
                size_slsqp = len(flux_opt_slsqp)
                for i in range(size_slsqp):
                    pvalue, rss_thres = model.goodness_of_fit(flux_opt_slsqp[i], alpha=0.05)
                    results.append((method_local, flux_opt_slsqp[i]))
                model.show_results(results, filename=result_path + f"_{Glc}_{Lac}.csv", format="csv")
                model.show_results(results, pool_size="off", checkrss="on")
            except IndexError:
                pass

end = time.localtime()
#duration = end - start
#print(duration)
