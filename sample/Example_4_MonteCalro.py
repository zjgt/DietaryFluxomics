﻿#-------------------------------------------------------------------------------
# Name:        mfapy example 4 An example of the Metropolis-Hastings algorithm
#              https://www.nature.com/articles/s41598-019-57146-8
#              Scientific Reports volume 10, Article number: 286 (2020)
# Author:      Fumio_Matsuda
#
# Created:     08/19/2020
# Copyright:   (c) Fumio_Matsuda 2018
# Licence:     MIT license
#-------------------------------------------------------------------------------
import os, sys, time
import numpy as numpy
import mfapy as mfapy
import scipy.stats as stats
import copy
import csv
import mkl
mkl.set_num_threads(1)# for Ryzen




def calcrsss(Rm_ind, parameters, func):
    """
    Low level function for residual sum of square calculation from mfapy.metabolicmodel.MetaboliModel.calc_rss

    Parameters
    ----------
    Rm_ind: list. vector of independent flux
    parameters: dic of parameters.
    func: function to calc MDV
    Returns
    -------
    m_ind: New vector of independent flux
    rss: rss of new independent flux
    acceptd: number of non-progressed steps

    """
    mkl.set_num_threads(1)
    Rm_initial = parameters['Rm_initial']
    stoichiometric_num = parameters['stoichiometric_num']
    reaction_num = parameters['reaction_num']
    reac_met_num = parameters['reaction_num']
    matrixinv = parameters['matrixinv']
    experiments = parameters['experiments']
    mdv_exp = numpy.array(parameters['mdv_exp'])
    mdv_use = parameters['mdv_use']
    covinv = parameters['covinv']
    lb = parameters['lb']
    ub = parameters['ub']
    lbi =parameters['lbi']
    ubi = parameters['ubi']
    df = parameters['df']
    rss = parameters['rss']

    if isinstance(func, dict):
        calmdv = func["calmdv"]
        diffmdv = func["diffmdv"]
    else:
        locals_dic = locals()
        exec(func, globals(), locals_dic)
        calmdv = locals_dic["calmdv"]
        diffmdv = locals_dic["diffmdv"]

    Length_ind = len(Rm_ind)

    pdf = stats.chi2.pdf(x = rss, df = df)
    accept = 0
    acceptd = 0
    hamatta = 0

    while accept < 1000:
        if hamatta > 100:
            return Rm_ind, rss, -1



        Rm_ind_next = Rm_ind[:]
        dame = 0
        for i in range(3):
        #for reac_num in range(Length_ind):
            reac_num = numpy.random.randint(0, Length_ind)
            for j in range(10):
                Rm_ind_next_temp = Rm_ind_next[:]
                perturbation =  (numpy.random.rand() - 0.5) * 2 * (ubi[reac_num]-lbi[reac_num])/100
                if lbi[reac_num] < Rm_ind_next[reac_num] + perturbation < ubi[reac_num]:
                    Rm_ind_next_temp[reac_num] = Rm_ind_next[reac_num] + perturbation
                    Rm = numpy.array(list(Rm_initial))
                    Rm[stoichiometric_num: reaction_num] = list(Rm_ind_next_temp)
                    tmp_r = numpy.dot(matrixinv, Rm)
                    temp = sum([1 for x in numpy.array(lb) - tmp_r if x > 0]) + sum([1 for x in tmp_r - numpy.array(ub) if x > 0])
                    if temp > 0:
                        continue
                    Rm_ind_next[reac_num] = Rm_ind_next[reac_num] + perturbation
                    break
            else:
                dame = dame + 1
        if dame == 3:
            print("hamatta")
            hamatta = hamatta + 1
            continue



        Rm = numpy.array(list(Rm_initial))
        Rm[stoichiometric_num: reaction_num] = list(Rm_ind_next)
        tmp_r = numpy.dot(matrixinv, Rm)

        mdv_original = list(tmp_r)

        for experiment in sorted(experiments.keys()):
            target_emu_list = experiments[experiment]['target_emu_list']
            mdv_carbon_sources = experiments[experiment]['mdv_carbon_sources']
            #
            mdv_original_temp, mdv_hash = calmdv(list(tmp_r), target_emu_list, mdv_carbon_sources)

            mdv_original.extend(mdv_original_temp)

        mdv = numpy.array([y for x, y in enumerate(mdv_original) if mdv_use[x] != 0])
        res = mdv_exp - mdv
        f = numpy.dot(res, numpy.dot(covinv, res))

        rss_next = f #+ sum


        pdf_next = stats.chi2.pdf(x = rss_next, df = df)
        #
        # If probabirity of next point is smaller than that of present point
        #
        if pdf_next < pdf:
            if numpy.random.rand() > pdf_next/pdf:
                    acceptd = acceptd + 1
                    accept = accept + 1
                    continue
        Rm_ind[:] = Rm_ind_next[:]
        rss = rss_next
        pdf = pdf_next
        accept = accept + 1

    return Rm_ind, rss, acceptd


if __name__ == '__main__':
    #
    #
    #
    #
    metabolicmodel = "Example_4_MCF7_model.txt"
    metabolicstate = "Example_4_flux_dist_MCF7.csv" # This file is used to set model constraints
    mdvfile1 = 'Example_4_C24_U_CITMAL.txt'
    mdvfile2 = 'Example_4_T24_U_CITMAL.txt'
    outputfilename = 'xample_4_output.csv'
    #
    # Load metabolic model from txt file to four dictionary
    #
    reactions, reversible, metabolites, target_fragments = mfapy.mfapyio.load_metabolic_model(metabolicmodel)
    #
    # Construct MetabolicModel instance
    #
    model = mfapy.metabolicmodel.MetabolicModel(reactions, reversible, metabolites, target_fragments)
    #
    # Configurations
    #
    model.set_configuration(callbacklevel = 0)
    model.set_configuration(iteration_max = 10000) # Maximal iternations in optimization
    model.set_configuration(number_of_repeat = 3) #Iteration in self.fitting_flux(method = 'deep') [SLSQP => LN_PRAXIS] * n
    model.set_configuration(ncpus = 8) #Number of local CPUs for Parallel computing
    #
    stdev = 0.015
    #
    # Load metabolite state from text file
    #
    state_dic = model.load_states(metabolicstate, format = 'csv')
    model.set_constraints_from_state_dict(state_dic)
    model.update()
    #
    # Generate instances of CarbonSource class from model
    #
    carbon_source1 = model.generate_carbon_source_templete()
    #
    # Set isotope labelling of carbon sources 1
    #
    carbon_source1 = model.generate_carbon_source_templete()
    carbon_source1.set_all_isotopomers('SubsCO2', [0.99, 0.01])
    carbon_source1.set_each_isotopomer('SubsGlc',{'#000000': 1.0}, correction = 'no')
    carbon_source1.set_each_isotopomer('SubsGln',{'#11111': 1.0}, correction = 'no')
    #
    #
    #
    fluxtemp, state = model.generate_state()
    recordini = []
    #
    # Creation of header
    #
    mdv = model.generate_mdv(fluxtemp, carbon_source1)
    tmp_r = ['', 'RSS']
    tmp_r.extend([id for (group, id) in model.vector["ids"]])
    recordini.append(tmp_r)


    record = recordini[:]
    for mdvfile in [mdvfile1, mdvfile2]:

        mdv_observed1 = model.load_mdv_data(mdvfile)
        model.set_experiment('ex1', mdv_observed1, carbon_source1)

        state, flux_opt1 = model.generate_initial_states(100, 1)
        method = "GN_CRS2_LM"
        state, RSS_bestfit, flux_opt1 = model.fitting_flux(method = method, flux = flux_opt1)
        for method in ["deep", "GN_CRS2_LM", "deep"]:
            start = time.time()
            state, RSS_bestfit, flux_opt1 = model.fitting_flux(method = method, flux = flux_opt1)
        #model.show_results([("test",flux_opt1)])


        rss = model.calc_rss(flux_opt1)
        df = mdv_observed1.get_number_of_measurement()
        sf = stats.chi2.sf(x = rss, df = df)

        print(mdvfile, "RSS:",rss, "p-value of chisq test", sf)
        if sf < 0.95:
            print("The metabolic model failed to overfit to MDV file", mdvfile)
            continue

        #
        # Preparation of parameters
        #

        Rm_ind = [flux_opt1[group][id]["value"] for (group, id) in model.vector['independent_flux']]
        lbi = [flux_opt1[group][id]["lb"] for (group, id) in model.vector['independent_flux']]
        ubi = [flux_opt1[group][id]["ub"] for (group, id) in model.vector['independent_flux']]
        stoichiometric_num = model.numbers['independent_start']
        reaction_num=  model.numbers['total_number']
        Rm_initial= model.vector["Rm_initial"]
        mdv_exp_original = list(model.vector["value"])
        mdv_std_original = list(model.vector["stdev"])
        mdv_use = list(model.vector["use"])
        for experiment in sorted(model.experiments.keys()):
            mdv_exp_original.extend(model.experiments[experiment]['mdv_exp_original'])
            mdv_std_original.extend(model.experiments[experiment]['mdv_std_original'])
            mdv_use.extend(model.experiments[experiment]['mdv_use'])
        mdv_exp = numpy.array([y for x, y in enumerate(mdv_exp_original) if mdv_use[x] != 0])
        spectrum_std = numpy.array([y for x, y in enumerate(mdv_std_original) if mdv_use[x] != 0])
        #
        # Covariance matrix
        #
        covinv = numpy.zeros((len(spectrum_std),len(spectrum_std)))
        for i, std in enumerate(spectrum_std):
            if std <= 0.0:
                print("Error in ", i, std)
            covinv[i,i] = 1.0/(std**2)
        #
        # Total n by 1000 steps progress of Metropolis-Hastings method
        #
        n = 5000
        for ti in range(n):
            #
            # Set parameters
            #
            parameters ={"reaction_num": model.numbers['total_number'],
                "stoichiometric_num": model.numbers['independent_start'],
                "matrixinv":model.matrixinv,
                "experiments":model.experiments,
                "mdv_exp":mdv_exp,
                "mdv_use":mdv_use,
                "covinv":covinv,
                "Rm_initial": model.vector["Rm_initial"],
                "lb" : copy.copy(model.vector["lb"]),
                "ub" : copy.copy(model.vector["ub"]),
                "reac_met_number" : model.numbers['reac_met_number'],
                "lbi" : lbi,
                "ubi" : ubi,
                "df" : df,
                "rss" : rss
            }
            #
            # 1000 steps progress of Metropolis-Hastings method. Please use joblib if you like.
            #
            Rm_ind_next, rss_next, acceptd = calcrsss(Rm_ind, parameters, model.calmdv_text)
            #
            # Show progress
            #
            print(mdvfile, str(ti), "RSS:",rss, "probability in chisq sdist", stats.chi2.pdf(x = rss, df = df), acceptd, df)
            #
            # acceptd = -1 means the metabolic state goes outside of lower and upper boundaries
            #
            if acceptd == -1:
                Rm_ind_next = Rm_ind
                rss_next = rss
                print("Failed to find proposal flux>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #
            # Check whether the metabolic state is inside of lower and upper boundaries
            #
            Rm = numpy.array(list(Rm_initial))
            Rm[stoichiometric_num: reaction_num] = list(Rm_ind)
            tmp_r =list(numpy.dot(model.matrixinv, Rm))
            lb = copy.copy(model.vector["lb"])
            ub = copy.copy(model.vector["ub"])
            temp = sum([1 for x in numpy.array(lb) - tmp_r if x > 0]) + sum([1 for x in tmp_r - numpy.array(ub) if x > 0])
            if temp > 0:
                Rm_ind_next = Rm_ind
                rss_next = rss
                print("flux is out of range       >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #
            # Moving to next step
            #
            Rm_ind = Rm_ind_next
            rss = rss_next
            #
            # Stored in record
            #
            Rm = numpy.array(list(Rm_initial))
            Rm[stoichiometric_num: reaction_num] = list(Rm_ind)
            tmp_r =list(numpy.dot(model.matrixinv, Rm))
            tmp_record = [mdvfile, rss]
            tmp_record.extend(tmp_r)
            record.append(tmp_record)

        with open(outputfilename, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(record)






