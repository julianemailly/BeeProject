'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import management_of_data_functions
import bout_functions


# Optimal route assessment  --------------------------------------------------------------


def optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality) : 
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
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  number_of_simulations = 10 #100
  number_of_bouts = 30
  number_of_bees = 1

  list_of_visitation_sequences = []

  matrix_of_bee_data = np.full((number_of_simulations*number_of_bouts, 2),None)

  best_quality_of_sim = np.zeros(number_of_simulations)
  best_route_sim = []

  i = 0


  # The simulations will be done without Q_Learning for the moment: 
  saved_use_Q_learning = bee_data.loc[0, "use_Q_learning"]
  saved_online_reinforcement = bee_data.loc[0, "online_reinforcement"]
  for ind in range (number_of_bees) : 
    bee_data.loc[ind, "use_Q_learning"] = False
    bee_data.loc[ind, "online_reinforcement"] = True

  for sim in range (number_of_simulations) : 

    probability_array_list = initial_probability_matrix_list

    for bout in range (number_of_bouts) : 

      management_of_data_functions.reboot_bee_data(bee_data)
      current_bout =  bout_functions.competitive_route(bout,array_geometry,probability_array_list,bee_data,optimal_route_quality,True,output_folder,number_of_bees=number_of_bees)

      #update variables
      optimal_route_quality = current_bout["optimal_route_quality"]

      matrix_of_bee_data[i,0] = sim
      matrix_of_bee_data[i,1] = current_bout["route_quality"][0]

      list_of_visitation_sequences.append(current_bout["sequences"])

      i = i + 1

    qualities_of_sim = matrix_of_bee_data[matrix_of_bee_data[:,0]==sim,1]
    best_quality_of_sim[sim] = np.max(qualities_of_sim)
    bout_of_best_quality = np.argmax(qualities_of_sim)
    best_route_sim.append(list_of_visitation_sequences[sim*number_of_bouts+bout_of_best_quality])

  # Assess the max quality attained in all sim
  sim_opt_route_quality = np.max(best_quality_of_sim)
  
  # Reformatting best_route_sim to be a numpy array
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
  best_route_sim = best_route_sim_array

  sim_reach_opti = best_route_sim[np.array(best_quality_of_sim) == sim_opt_route_quality]
  all_best_routes = np.unique(sim_reach_opti,axis=0)

  proportion_of_sim_with_opt_qual = 100* np.sum(best_quality_of_sim==sim_opt_route_quality) / len(best_quality_of_sim)

  if not silent_sim : 
    print("Out of "+str(number_of_simulations)+" simulations in "+str(array)+", "+str(proportion_of_sim_with_opt_qual)+"% reached the maximum quality of "+str(sim_opt_route_quality)+". Setting this value as optimal route quality.")
    print("A total of "+str(len(all_best_routes))+" routes had this quality. They were the following :")
    print(all_best_routes)

  # Do not forget to put back the original use_Q_learning and online_reinforcement paramters in bee_data: 
  for ind in range (number_of_bees) : 
    bee_data.loc[ind, "use_Q_learning"] = saved_use_Q_learning
    bee_data.loc[ind, "online_reinforcement"] = saved_online_reinforcement

  return(sim_opt_route_quality)


def optimal_route_assessment_2_ind(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality) : 
  """
  Description:
    Makes an assessment of the optimal route quality for two foraging individuals of an array by releasing 100 pairs of bees for 30 bout each. Retrieves the best route quality.
    This is an approximation of the solution of the travelling salesman problem.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  number_of_simulations = 10 #100
  number_of_bouts = 30
  number_of_bees = 2

  if len(bee_data.index) < number_of_bees : 
    if not silent_sim : 
      print("Assessing the optimal route for two individuals is not possible: number_of_bees < 2")
    return(None)

  else : 
    list_of_visitation_sequences = []
    list_of_resources_taken = []

    matrix_of_bee_data = np.full((number_of_simulations*number_of_bouts, 2),None)

    best_quality_of_sim = np.zeros(number_of_simulations)
    best_route_sim = []

    i = 0

    print(np.shape(initial_probability_matrix_list))

    # The simulations will be done without Q_Learning for the moment: 
    saved_use_Q_learning = bee_data.loc[0,"use_Q_learning"]
    saved_online_reinforcement = bee_data.loc[0,"online_reinforcement"]
    for ind in range (number_of_bees) : 
      bee_data.loc[ind, "use_Q_learning"] = True
      bee_data.loc[ind, "online_reinforcement"] = True

    for sim in range (number_of_simulations) : 

      probability_array_list = initial_probability_matrix_list

      for bout in range (number_of_bouts) : 
        management_of_data_functions.reboot_bee_data(bee_data)
        current_bout =  bout_functions.competitive_route(bout,array_geometry,probability_array_list,bee_data,optimal_route_quality,True,output_folder,number_of_bees=number_of_bees)

        #update variables
        optimal_route_quality = current_bout["optimal_route_quality"]

        matrix_of_bee_data[i,0] = sim
        matrix_of_bee_data[i,1] = current_bout["route_quality"][0] + current_bout["route_quality"][1] # there are gonna be duplications of the same number but doesn't matter

        list_of_visitation_sequences.append(current_bout["sequences"])
        i = i + 1

        list_of_resources_taken.append(current_bout["list_of_bout_resources"])

      qualities_of_sim = matrix_of_bee_data[matrix_of_bee_data[:,0]==sim,1]
      best_quality_of_sim[sim] = np.max(qualities_of_sim)
      bout_of_best_quality = int(np.ceil(np.argmax(qualities_of_sim)/2)) # need to account for the repetitions in the bouts due to the nb of bees
      best_route_sim.append(list_of_visitation_sequences[sim*number_of_bouts+bout_of_best_quality])


    # Assess the max quality attained in each sim
    sim_opt_route_quality = np.max(best_quality_of_sim)
    print(sim)

    # Reformatting best_route_sim to be a numpy array
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
    best_route_sim = best_route_sim_array

    sim_reach_opti = best_route_sim[np.array(best_quality_of_sim) == sim_opt_route_quality]
    all_best_routes = np.unique(sim_reach_opti,axis=0)

    proportion_of_sim_with_opt_qual = 100* np.sum(best_quality_of_sim==sim_opt_route_quality) / len(best_quality_of_sim)


    if not silent_sim : 
      print("Out of "+str(number_of_simulations)+" simulations in "+str(array)+", "+str(proportion_of_sim_with_opt_qual)+"% reached the maximum quality of "+str(sim_opt_route_quality)+". Setting this value as optimal route quality.")
      print("A total of "+str(len(all_best_routes))+" routes had this quality. They were the following :")
      print(all_best_routes)

    # Do not forget to put back the original use_Q_learning and online_reinforcement paramters in bee_data: 
    for ind in range (number_of_bees) : 
      bee_data.loc[ind, "use_Q_learning"] = saved_use_Q_learning
      bee_data.loc[ind, "online_reinforcement"] = saved_online_reinforcement

    return(sim_opt_route_quality)


def sim_detection_optimal_route(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality) : 
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
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if not silent_sim : 
    print("Checking on array folder : "+output_folder)
  known_files =  np.array(os.listdir(output_folder))
  number_of_files = len(known_files)

  if (number_of_files>0) and ("optimal_route.csv" in known_files) : 
    if not silent_sim: 
      print("There is a optimal_route.csv file available for this array.")
    data_file = pd.read_csv(output_folder+'\\optimal_route.csv')
    optimal_route = data_file.iloc[0,0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality)
    # Save it in a file
    pd.DataFrame({"optimal_route":[optimal_route]}).to_csv(path_or_buf = output_folder+'\\optimal_route.csv', index = False)

  return(optimal_route)


def sim_detection_optimal_route_2_ind(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality) : 
  """
  Description:
    Attempts to retrieve an existing optimal route for two individuals file for the array tested. Otherwise, calls optimal_route_assessment_2_ind.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix_list: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if not silent_sim : 
    print("Checking on array folder : "+output_folder)
  known_files =  np.array(os.listdir(output_folder))
  number_of_files = len(known_files)

  if (number_of_files>0) and ("optimal_route_2_ind.csv" in known_files) : 
    if not silent_sim: 
      print("There is a optimal_route_2_ind.csv file available for this array.")
    data_file = pd.read_csv(output_folder+'\\optimal_route_2_ind.csv')
    optimal_route = data_file.iloc[0,0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route for 2 individuals found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment_2_ind(array,array_geometry,bee_data,initial_probability_matrix_list,output_folder,silent_sim,optimal_route_quality)
    # Save it in a file
    pd.DataFrame({"optimal_route":[optimal_route]}).to_csv(path_or_buf = output_folder+'\\optimal_route_2_ind.csv', index = False)

  return(optimal_route)


#test
"""

import geometry_functions
import spatial_array_generation_and_manipulation_functions



alpha_pos = 0.5
alpha_neg = 0.2
gamma_QL = 0
online_reinforcement = True
cost_of_flying = False
learning_factor = 1.5
abandon_factor = 0.75
leave_after_max_fail = True

different_experience_simulation = False
starting_bout_for_naive=[0,0]

online_reinforcement = True
use_Q_learning = True
dynamic_beta=True
beta_QL=2
beta_QL_vector=[2 for k in range (100)]
array_number = 1

current_pos = 1
previous_pos = 0
number_of_bees = 1
number_of_flowers_plus_nest = 10

optimal_route_quality = 0
silent_sim = False

array = "test_name_array"
array_number = 1
array_info = {'environment_type':'generate', 'number_of_flowers' : number_of_flowers_plus_nest-1, 'number_of_patches' : 1, 'patchiness_index' : 0, 'env_size' : 500, 'flowers_per_patch' : None }
array_geometry, array_info_new, array_folder = spatial_array_generation_and_manipulation_functions.create_environment (array_info, array_number,True,False)
print('array geometry ',array_geometry)
param_tracking = {"starting_bout_for_naive":starting_bout_for_naive,"different_experience_simulation":different_experience_simulation,"number_of_resources_foraged" : 0,"bout_finished": False,"distance_travelled":0.}
param_indiv = {"max_fails":2,"route_compare":False,"max_distance_travelled": 10000,"max_crop":5,"probability_of_winning":0.5,"dynamic_beta":dynamic_beta,"beta_QL_vector":beta_QL_vector,"leave_after_max_fail": leave_after_max_fail, "max_fails":2,"forbid_reverse_vector": True,"allow_nest_return": False,"beta_QL": beta_QL,"alpha_pos":alpha_pos,"alpha_neg":alpha_neg,"gamma_QL":gamma_QL,"use_Q_learning":use_Q_learning,"online_reinforcement": online_reinforcement,"cost_of_flying":cost_of_flying,"learning_factor":learning_factor,"abandon_factor":abandon_factor}
bee_data = management_of_data_functions.initialize_bee_data(number_of_bees,param_tracking,param_indiv)
dist_factor = 2 
initial_probability_matrix_list = geometry_functions.initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees)
current_working_directory = os.getcwd()
output_folder = current_working_directory+'\\Arrays'
optimal_route_quality = 0

optimal_route=sim_detection_optimal_route_2_ind(array,array_geometry,bee_data,initial_probability_matrix_list,array_folder,silent_sim,optimal_route_quality) 

"""
