#-------------------------------------------------------------------------------
# Name:        Flux2_grid_search_A549.py
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

    source_dir = 'Flux2_MDV_A549_Files/'
    dest_dir = 'Flux2_Results/'
    files = os.listdir(source_dir)
    files = np.sort(files)

    reactions, reversible, metabolites, target_fragments = mfapy.mfapyio.load_metabolic_model(
        "Flux2_model_A549.txt")
    model = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, target_fragments)
    model.set_configuration(callbacklevel=4)
    model.set_configuration(iteration_max=1000000)  # Maximal iterations in optimization
    model.set_configuration(number_of_repeat=3)
    model.set_configuration(ncpus=20)  # Number of local CPUs for Parallel python
    state_dic = model.load_states('Flux2_status_A549.csv', format='csv')
    model.set_constraints_from_state_dict(state_dic)
    model.update()
    for file in files:
        if file.endswith('.DS_Store'):
            continue
        else:
            print(file)
            filename = file.split('/')[-1].split('.')[0]
            result_path = dest_dir + filename
            for k in range(11):
                Glc12 = 0.30 + 0.01 * k
                Glc13 = 1.00 - Glc12
                Glc100 = int(100 * Glc12)

                carbon_source1 = model.generate_carbon_source_template()
                carbon_source1.set_all_isotopomers('SubsCO2', [1.0, 0.0], correction='no')
                carbon_source1.set_each_isotopomer('SubsGlc', {'#000000': Glc12, '#111111': Glc13},
                                                   correction='no')
                carbon_source1.set_each_isotopomer('SubsGln', {'#00000': 1.}, correction='no')
                #carbon_source1.set_each_isotopomer('SubsSer', {'#000': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsAla', {'#000': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsLac', {'#000': 1.0, '#111': 0.0}, correction='no')
                carbon_source1.set_each_isotopomer('SubsR5P', {'#00000': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsAcCOAmit', {'#00': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsAsp', {'#0000': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsGly', {'#000': 1.}, correction='no')
                carbon_source1.set_each_isotopomer('SubsGlycogen', {'#000000': 1.}, correction='no')
                mdv_observed1 = model.load_mdv_data(source_dir + file)
                mdv_observed1.set_std(0.015, method='absolute')
                model.set_experiment('ex1', mdv_observed1, carbon_source1)  #
                method_global = "GN_CRS2_LM"
                method_local = "SLSQP"
                state1, flux_opt1 = model.generate_initial_states(20000, 20, method="parallel")
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
                    model.show_results(results, filename=result_path + f'_{Glc100}_{k}.csv', format="csv")
                    model.show_results(results, pool_size="off", checkrss="on")
                except IndexError:
                    pass


end = time.localtime()


