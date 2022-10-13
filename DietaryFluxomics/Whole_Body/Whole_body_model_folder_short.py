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


if __name__ == '__main__':
    #
    # Construction of metabolic model
    #
    reactions, reversible, metabolites, target_fragments = mfapy.mfapyio.load_metabolic_model("Whole_body_models/Whole_body_model_updated_v22.txt")
    model = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, target_fragments)
    #
    # Configurations
    #
    model.set_configuration(callbacklevel = 4)
    model.set_configuration(iteration_max = 100000) # Maximal iternations in optimization
    model.set_configuration(number_of_repeat = 4) #Iteration in self.fitting_flux(method = 'deep') [SLSQP => LN_PRAXIS] * n
    model.set_configuration(ncpus = 10) #Number of local CPUs for Parallel python
    #
    # Loading metabolite state from text file to constrain flux vector
    #
    state_dic = model.load_states("Whole_body_models/Whole_body_status_ub_updated_v25.csv", format = 'csv')
    model.set_constraints_from_state_dict(state_dic)
    model.update()
    #
    # Generation of CarbonSource instance
    #
    #
    # Isotope labelling information of carbon sources 1
    #
    carbon_source1 = model.generate_carbon_source_template()
    carbon_source1.set_all_isotopomers('SubsCO2', [0.99, 0.01])
    carbon_source1.set_each_isotopomer('SubsGlcPeri', {'#000000': 0.25, '#111111': 0.75}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsGlcCn', {'#000000': 0.65, '#111111': 0.35}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsGln', {'#00000': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsSer', {'#000': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsAla', {'#000': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsLacPeri', {'#000': 0.25, '#111': 0.75}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsLacCn', {'#000': 0.40, '#111': 0.60}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsR5P', {'#00000': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsAcCOAmit', {'#00': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsGly', {'#000': 1.}, correction='yes')
    carbon_source1.set_each_isotopomer('SubsUridine', {'#000000000': 1.}, correction='yes')
    #
    # Generation of CarbonSource instance
    #
    #
    # Loading of measured MDV data 1
    #
    start = time.time()
    source_dir = 'D:/mfapyt/sample/Whole_body_models/Whole_body_MDV_files_short/'
    dest_dir = 'D:/mfapyt/sample/Whole_body_models/Whole_body_modeling_results_short/'
    files = os.listdir(source_dir)
    for file in files:
        print(file)
        result_path = dest_dir + file
        mdv_observed1 = model.load_mdv_data(source_dir + file)
        mdv_observed1.set_std(0.015, method='absolute')
        #
        # Addition of labeling experiment 1
        #
        start = time.time()
        model.set_experiment('ex1', mdv_observed1, carbon_source1)  #
        #
        #
        # method_global = "GN_CRS2_LM"
        method_global = "LN_SBPLX"
        # methods = ["SLSQP", "COBYLA", "LN_COBYLA", "LN_BOBYQA", "LN_NEWUOA", "LN_PRAXIS", "LN_SBPLX",
        #           "LN_NELDERMEAD", "GN_CRS2_LM", "deep"]
        state1, flux_opt1 = model.generate_initial_states(1000, 10, method="parallel")
        results = [('template', flux_opt1[0])]
        num = len(flux_opt1)

        state, RSS_bestfit, flux_opt1 = model.fitting_flux(method=method_global, flux=flux_opt1)
        num = len(flux_opt1)
        for i in range(num):
            results.append((method_global, flux_opt1[i]))
        model.show_results(results, filename=result_path + '.csv', format="csv")
        print(time.time() - start)
    now = time.time()
    print(now-start)