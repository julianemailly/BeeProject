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


def optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality,number_of_bees) : 
  """
  Description:
    Makes an assessment of the optimal route quality of an array by releasing 100 bees for 30 bout each. Retrieves the best route quality.
    This is an approximation of the solution of the travelling salesman problem.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
    number_of_bees: number of bees for which the optimal quality will be assessed
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement
  """
  number_of_simulations = 15 #100
  number_of_bouts = 30

  if len(bee_data.index) < number_of_bees : 
    if not silent_sim : 
      print("Assessing the optimal route for two individuals is not possible: number of bees < parameters.number_of_bees")
    return(None)

  else : 

    list_of_visitation_sequences = []

    matrix_of_bee_data = np.full((number_of_simulations*number_of_bouts, 2),None)

    best_quality_of_sim = np.zeros(number_of_simulations)
    best_route_sim = []

    i = 0


    # The simulations will be done without Q_Learning for the moment and use_dynamic_beta will be forcedly disactivated: 
    saved_use_Q_learning = copy.deepcopy(bee_data.loc[:, "use_Q_learning"])
    saved_use_online_reinforcement = copy.deepcopy(bee_data.loc[:, "use_online_reinforcement"])
    saved_use_dynamic_beta = copy.deepcopy(bee_data.loc[:, "use_dynamic_beta"])
    saved_beta_vector = copy.deepcopy(bee_data.loc[:, "beta_vector"])

    bee_data.loc[:, "use_Q_learning"] = False
    bee_data.loc[:, "use_online_reinforcement"] = True
    for sim in range (number_of_simulations) : 

      probability_array_list = initial_probability_matrix_list

      for bout in range (number_of_bouts) : 

        management_of_data_functions.reboot_bee_data(bee_data)
        current_bout =  bout_functions.simulate_bout(bout,array_geometry,probability_array_list,bee_data,optimal_route_quality,True,output_folder,number_of_bees=number_of_bees)

        #update variables
        optimal_route_quality = current_bout["optimal_route_quality"]

        matrix_of_bee_data[i,0] = sim
        matrix_of_bee_data[i,1] = np.sum(current_bout["route_quality"]) #will summate the route qualities of each bee

        list_of_visitation_sequences.append(current_bout["sequences"])

        i = i + 1

      qualities_of_sim = matrix_of_bee_data[matrix_of_bee_data[:,0]==sim,1]
      best_quality_of_sim[sim] = np.max(qualities_of_sim)
      bout_of_best_quality =  int(np.ceil(np.argmax(qualities_of_sim)/number_of_bees)) # need to account for the repetitions in the bouts due to the nb of bees
      best_route_sim.append(list_of_visitation_sequences[sim*number_of_bouts+bout_of_best_quality])

    # Assess the max quality reached in all sim
    sim_opt_route_quality = np.max(best_quality_of_sim)

    # Reformatting best_route_sim to be a numpy array
    best_route_sim = reformatting_best_route_sim(best_route_sim,number_of_simulations)

    sim_reach_opti = best_route_sim[np.array(best_quality_of_sim) == sim_opt_route_quality]
    all_best_routes = np.unique(sim_reach_opti,axis=0)

    proportion_of_sim_with_opt_qual = 100* np.sum(best_quality_of_sim==sim_opt_route_quality) / len(best_quality_of_sim)

    if not silent_sim : 
      print("Out of "+str(number_of_simulations)+" simulations in "+str(array)+", "+str(proportion_of_sim_with_opt_qual)+"% reached the maximum quality of "+str(sim_opt_route_quality)+". Setting this value as optimal route quality.")
      print("A total of "+str(len(all_best_routes))+" routes had this quality. They were the following :")
      print(all_best_routes)

    # Do not forget to put back the original use_Q_learning and use_online_reinforcement paramters in bee_data: 
    bee_data.loc[:, "use_Q_learning"] = saved_use_Q_learning
    bee_data.loc[:, "use_online_reinforcement"] = saved_use_online_reinforcement
    bee_data.loc[:, "use_dynamic_beta"] = saved_use_dynamic_beta
    bee_data.loc[:, "beta_vector"] = saved_beta_vector
    return(sim_opt_route_quality)




def retrieve_optimal_route(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality,number_of_bees) : 
  """
  Description:
    Attempts to retrieve an existing optimal route file for the array tested. Otherwise, calls optimal_route_assessment.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
    number_of_bees: number of bees for which the optimal quality will be assessed
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if not silent_sim : 
    print("Checking on array folder : "+output_folder)
  known_files =  np.array(os.listdir(output_folder))
  number_of_files = len(known_files)
  name_of_csv_file = "optimal_route_"+str(number_of_bees)+"_ind.csv"

  if (number_of_files>0) and (name_of_csv_file in known_files) : 
    if not silent_sim: 
      print("There is a " +name_of_csv_file+" file available for this array.")
    data_file = pd.read_csv(output_folder+'\\'+name_of_csv_file)
    optimal_route = data_file.iloc[0,0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route for "+str(number_of_bees)+" individuals found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality,number_of_bees)
    # Save it in a file
    pd.DataFrame({"optimal_route":[optimal_route]}).to_csv(path_or_buf = output_folder+'\\'+name_of_csv_file, index = False)

  return(optimal_route)