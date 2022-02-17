'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import copy
import management_of_data_functions
import bout_functions


# Optimal route assessment  --------------------------------------------------------------

def reformatting_best_route_sim(best_route_sim,number_of_simulations):
  """
  Description: 
    Reformat best_route_sim to be an array
  Inputs:
    best_route_sim: list of length number_of_simulations containing the best route (or list of routes if several indivuals) in each simulation
  Outputs:
    best_route_sim but reformated to be an numpy array
  """
  max_nrow,max_ncol = 0,0
  for sim in range (number_of_simulations) : 
    nrow,ncol = np.shape(best_route_sim[sim])
    if nrow > max_nrow : 
      max_nrow = nrow
    if ncol > max_ncol : 
      max_ncol = ncol

  best_route_sim_array = np.full((number_of_simulations,max_nrow,max_ncol),0)
  for sim in range (number_of_simulations) : 
    nrow,ncol = np.shape(best_route_sim[sim])
    best_route_sim_array[sim,:nrow,:ncol] = best_route_sim[sim]
  return(best_route_sim_array)


def optimal_route_assessment(array_geometry,bee_data,bee_info,array_folder,silent_sim,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_bees) : 
  """
  Description:
    Makes an assessment of the optimal route quality of an array by releasing 100 bees for 30 bout each. Retrieves the best route quality.
    This is an approximation of the solution of the travelling salesman problem.
  Inputs:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bbee_data: numpy array of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    array_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
    number_of_bees: number of bees for which the optimal quality will be assessed
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement
  """
  number_of_simulations = 10 #100
  number_of_bouts = 30

  if np.shape(bee_data)[0] < number_of_bees : 
    if not silent_sim : 
      print("Assessing the optimal route for two individuals is not possible: number of bees < parameters.number_of_bees")
    return(None)

  else : 

    # The simulations will be done without Q_Learning for the moment and use_dynamic_beta will be forcedly disactivated: 
    saved_use_Q_learning = copy.deepcopy(bee_info.loc[:, "use_Q_learning"])
    saved_use_online_reinforcement = copy.deepcopy(bee_info.loc[:, "use_online_reinforcement"])
    saved_use_dynamic_beta = copy.deepcopy(bee_info.loc[:, "use_dynamic_beta"])
    saved_beta_vector = copy.deepcopy(bee_info.loc[:, "beta_vector"])

    bee_info.loc[:, "use_Q_learning"] = False
    bee_info.loc[:, "use_online_reinforcement"] = True

    best_qual_for_all_sim = 0 # best route quality so far, will be updated at the end of each simulation
    number_of_sim_having_the_best_qual = 1 # number of bouts having the best quality

    initial_probability_matrix_list = management_of_data_functions.initialize_probability_matrix_list(array_geometry,bee_info["dist_factor"][0],number_of_bees,bee_info["allow_nest_return"])

    for sim in range (number_of_simulations) : 

      probability_array_list = copy.deepcopy(initialize_probability_matrix_list)

      best_qual_for_this_sim = 0

      for bout in range (number_of_bouts) : 

        management_of_data_functions.reboot_bee_data(bee_data)
        current_bout =  bout_functions.simulate_bout(bout,array_geometry,probability_array_list,bee_data,bee_info,optimal_route_quality_1_ind,optimal_route_quality_2_ind,True,array_folder,number_of_bees=number_of_bees)

        #update variables
        optimal_route_quality_1_ind = current_bout["optimal_route_quality_1_ind"] # useful to launch the next bout
        optimal_route_quality_2_ind = current_bout["optimal_route_quality_2_ind"] # same

        optimal_route_quality_for_every_bee = np.sum(current_bout["route_quality"]) # allow to compute groupe quality without specifying the number of bees

        if optimal_route_quality_for_every_bee > best_qual_for_this_sim : 
          best_qual_for_this_sim = optimal_route_quality_for_every_bee 


      if best_qual_for_all_sim == best_qual_for_this_sim : 
        number_of_sim_having_the_best_qual +=1


      if best_qual_for_all_sim < best_qual_for_this_sim : 
        best_qual_for_all_sim = best_qual_for_this_sim
        number_of_sim_having_the_best_qual = 1

    proportion_of_sim_with_opt_qual = 100* number_of_sim_having_the_best_qual / number_of_simulations

    if not silent_sim : 
      print("Out of "+str(number_of_simulations)+" simulations, "+str(proportion_of_sim_with_opt_qual)+"% reached the maximum quality of "+str(best_qual_for_all_sim)+". Setting this value as optimal route quality.")

    # Do not forget to put back the original use_Q_learning and use_online_reinforcement paramters in bee_data: 
    bee_info.loc[:, "use_Q_learning"] = saved_use_Q_learning
    bee_info.loc[:, "use_online_reinforcement"] = saved_use_online_reinforcement
    bee_info.loc[:, "use_dynamic_beta"] = saved_use_dynamic_beta
    bee_info.loc[:, "beta_vector"] = saved_beta_vector
    return(best_qual_for_all_sim)




def retrieve_optimal_route(array_geometry,bee_data,bee_info,array_folder,silent_sim,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_bees) : 
  """
  Description:
    Attempts to retrieve an existing optimal route file for the array tested. Otherwise, calls optimal_route_assessment.
  Inputs:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: numpy array of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    array_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
    number_of_bees: number of bees for which the optimal quality will be assessed
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if number_of_bees <2: 
    optimal_route_quality_2_ind = None
  if not silent_sim : 
    print("Checking on array folder : "+array_folder)
  known_files =  np.array(os.listdir(array_folder))
  number_of_files = len(known_files)
  name_of_csv_file = "optimal_route_"+str(number_of_bees)+"_ind.csv"

  if (number_of_files>0) and (name_of_csv_file in known_files) : 
    if not silent_sim: 
      print("There is a " +name_of_csv_file+" file available for this array.")
    data_file = pd.read_csv(array_folder+'\\'+name_of_csv_file)
    optimal_route = data_file.iloc[0,0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route for "+str(number_of_bees)+" individuals found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment(array_geometry,bee_data,bee_info,array_folder,silent_sim,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_bees)
    # save it in a file
    pd.DataFrame({"optimal_route":[optimal_route]}).to_csv(path_or_buf = array_folder+'\\optimal_route_'+str(number_of_bees)+'_ind.csv', index = False)

  return(optimal_route)