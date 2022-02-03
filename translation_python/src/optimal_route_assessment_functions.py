'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import spatial_array_generation_and_manipulation_functions
import management_of_data_functions


# Optimal route assessment  --------------------------------------------------------------


def optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix,silent_sim=False,optimal_route_quality) : 
    """
  Description:
    Makes an assessment of the optimal route quality of an array by releasing 100 bees for 30 bout each. Retrieves the best route quality.
    This is an approximation of the solution of the travelling salesman problem.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  number_of_simulations = 100
  number_of_bouts = 30
  number_of_bees = 1

  list_of_visitation_sequences = np.empty(number_of_simulations*number_of_bouts)

  matrix_of_bee_data = np.zeros((number_of_simulations*number_of_bouts*number_of_bees, 5))
  for i in range (number_of_simulations) : 
    for j in range (nulber_of_bouts) : 
      for k in range (number_of_bees) : 
        matrix_of_bee_data[i+j+k,0] = i
        matrix_of_bee_data[i+j+k,1] = j
        matrix_of_bee_data[i+j+k,2] = k

  best_quality_of_sim = np.zeros(number_of_simulations)
  best_route_sim = []

  i = 0
  for sim in range (number_of_simulations) : 

    probability_array = initial_probability_matrix

    for bout in (number_of_bouts) : 
      bee_data = management_of_data_functions.reboot_bee_data(bee_data)
      current_bout =  competitive_route(bout,array_geometry,probability_array,bee_data[0,:],optimal_route_quality,silent_sim=True,use_Q_learning=False)

      #update variables
      probability_array = current_bout["learning"]
      bee_data = current_bout["bee_data"]
      optimal_route_quality = current_bout["optimal_route_quality"]
      matrix_of_bee_data[i,3] = bee_data["number_of_resources_foraged"][0]
      matrix_of_bee_data[i,4] = current_bout["quality"][0]

      list_of_visitation_sequences[sim*number_of_bouts+bout] = current_bout["sequences"]

      i = i + number_of_bees

    qualities_of_sim = matrix_of_bee_data[matrixOfBeeData[:,0]==sim,4]
    best_quality_of_sim[sim] = np.max(qualities_of_sim)
    bout_of_best_quality = np.argmax(qualities_of_sim)
    best_route_sim.append(np.concatenate((list_of_visitation_sequences[sim*number_of_bouts+bout_of_best_quality],[0])))

  # Assess the max quality attained in each sim
  sim_opt_route_quality = np.max(best_quality_of_sim)

  sim_reach_opti = best_route_sim[best_quality_of_sim == sim_opt_route_quality]
  all_best_routes = np.unique(sim_reach_opti)

  proportion_of_opt_route = 100* len(best_quality_of_sim[best_quality_of_sim==np.max(sim_opt_route_quality)]) / len(sim_opt_route_quality)


  if not silent_sim : 
    print("Out of "+str(number_of_simulations)+" simulations in "+str(array)+", "+str(proportion_of_opt_route)+"% reached the maximum quality of "+str(sim_opt_route_quality)+". Setting this value as optimal route quality.")
    print("A total of "+str(len(allBestRoutes))+" routes had this quality. They were the following :")
    print(all_best_routes)

  return(sim_opt_route_quality)


def optimal_route_assessment_2_ind(array,array_geometry,bee_data,initial_probability_matrix,silent_sim=False,optimal_route_quality) : 
    """
  Description:
    Makes an assessment of the optimal route quality for two foraging individuals of an array by releasing 100 pairs of bees for 30 bout each. Retrieves the best route quality.
    This is an approximation of the solution of the travelling salesman problem.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  number_of_simulations = 100
  number_of_bouts = 30
  number_of_bees = 2

  list_of_visitation_sequences = np.empty(number_of_simulations*number_of_bouts)
  list_of_resources_taken = np.empty(number_of_simulations*number_of_bouts)

  matrix_of_bee_data = np.zeros((number_of_simulations*number_of_bouts*number_of_bees, 5))
  for i in range (number_of_simulations) : 
    for j in range (nulber_of_bouts) : 
      for k in range (number_of_bees) : 
        matrix_of_bee_data[i+j+k,0] = i
        matrix_of_bee_data[i+j+k,1] = j
        matrix_of_bee_data[i+j+k,2] = k

  best_quality_of_sim = np.zeros(number_of_simulations)
  best_route_sim = []

  i = 0
  j=0

  for sim in range (number_of_simulations) : 

    probability_array = initial_probability_matrix

    for bout in (number_of_bouts) : 
      bee_data = management_of_data_functions.reboot_bee_data(bee_data)
      current_bout =  competitive_route(bout,array_geometry,probability_array,bee_data[0,:],optimal_route_quality,silent_sim=True,use_Q_learning=False)

      #update variables
      probability_array = current_bout["learning"]
      bee_data = current_bout["bee_data"]
      optimal_route_quality = current_bout["optimal_route_quality"]
      matrix_of_bee_data[i,3] = bee_data["number_of_resources_foraged"][0]
      matrix_of_bee_data[i,4] = current_bout["quality"][0] + current_bout["quality"][1] # there are gonna be duplications of the same number but doesn't matter

      list_of_visitation_sequences[sim*number_of_bouts+bout] = current_bout["sequences"]
      i = i + number_of_bees

      list_of_resources_taken[j] = current_bout["list_of_bout_resources"]
      j=j+1

    qualities_of_sim = matrix_of_bee_data[matrixOfBeeData[:,0]==sim,4]
    best_quality_of_sim[sim] = np.max(qualities_of_sim)
    bout_of_best_quality = int(np.ceil(np.argmax(qualities_of_sim)/2)) # need to account for the repetitions in the bouts due to the nb of bees
    best_route_sim.append(np.concatenate((list_of_visitation_sequences[sim*number_of_bouts+bout_of_best_quality],[0])))

  # Assess the max quality attained in each sim
  sim_opt_route_quality = np.max(best_quality_of_sim)

  sim_reach_opti = best_route_sim[best_quality_of_sim == sim_opt_route_quality]
  all_best_routes = np.unique(sim_reach_opti)

  proportion_of_opt_route = 100* len(best_quality_of_sim[best_quality_of_sim==np.max(sim_opt_route_quality)]) / len(sim_opt_route_quality)


  if not silent_sim : 
    print("Out of "+str(number_of_simulations)+" simulations in "+str(array)+", "+str(proportion_of_opt_route)+"% reached the maximum quality of "+str(sim_opt_route_quality)+". Setting this value as optimal route quality.")
    print("A total of "+str(len(allBestRoutes))+" routes had this quality. They were the following :")
    print(all_best_routes)

  return(sim_opt_route_quality)


def sim_detection_optimal_route(array,array_geometry,bee_data,initial_probability_matrix,output_folder,silent_sim=False,optimal_route_quality) : 
  """
  Description:
    Attempts to retrieve an existing optimal route file for the array tested. Otherwise, calls optimal_route_assessment.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if not silent_sim : 
    print("Checking on arrayFolder : "+outputFolder)
  known_files =  np.array(os.listdir(output_folder))
  number_of_files = len(known_files)

  if (number_of_files>0) and ("optimal_route.csv" in known_files) : 
    if not silent_sim: 
      print("There is a optimal_route.csv file available for this array.")
    data_file = pd.read_csv(output_folder+'\\optimal_route.csv', header = True)
    optimal_route = data_file[0][0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix,silent_sim,optimal_route_quality)
    # Save it in a file
    pd.DataFrame({"optimal_route":optimal_route}).to_csv(path_or_buf = output_folder+'\\optimal_route.csv', index = False)

  return(optimal_route)

def sim_detection_optimal_route_2_ind(array,array_geometry,bee_data,initial_probability_matrix,output_folder,silent_sim=False,optimal_route_quality) : 
  """
  Description:
    Attempts to retrieve an existing optimal route for two individuals file for the array tested. Otherwise, calls optimal_route_assessment_2_ind.
  Inputs:
    array: str, name of the array
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
    initial_probability_matrix: square matrix of size (number_of_flowers+1) giving the probabilty of going from one flower to another (including the nest) at the beginning of the simulation
    output_folder: string, path of the output folder
    silent_sim: boolean, if True prevents the prints
    optimal_route_quality: helps updtaing the current optimal route quality of the bee
  Outputs:
    optimal_route: float, optimal route quality for one individual in a given environement

  """
  if not silent_sim : 
    print("Checking on arrayFolder : "+outputFolder)
  known_files =  np.array(os.listdir(output_folder))
  number_of_files = len(known_files)

  if (number_of_files>0) and ("optimal_route_2_ind.csv" in known_files) : 
    if not silent_sim: 
      print("There is a optimal_route_2_ind.csv file available for this array.")
    data_file = pd.read_csv(output_folder+'\\optimal_route_2_ind.csv', header = True)
    optimal_route = data_file[0][0]
  else : 
    # If not found in the memory, assess the optimal route quality
    if not silent_sim : 
      print("No data on optimal route for 2 individuals found. Assessing optimal route with simulations.")
    optimal_route = optimal_route_assessment_2_ind(array,array_geometry,bee_data,initial_probability_matrix,silent_sim,optimal_route_quality)
    # Save it in a file
    pd.DataFrame({"optimal_route":optimal_route}).to_csv(path_or_buf = output_folder+'\\optimal_route_2_ind.csv', index = False)

  return(optimal_route)

#test
"""
import geometry_functions
array = "test_name_array"
array_number = 1
array_info = {'environment_type':'generate', 'number_of_resources' : 5, 'number_of_patches' : 1, 'patchiness_index' : 0, 'env_size' : 500, 'flowers_per_patch' : None }
array_geometry, array_info_new, array_folder = spatial_array_generation_and_manipulation_functions.create_environment (array_info, array_number)
print('array geometry ',array_geometry)
bee_data = 
dist_factor = 2 
number_of_bees = 1
initial_probability_matrix = geometry_functions.initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees)[0]
current_working_directory = os.getcwd()
output_folder = current_working_directory+'\\Outputs'
optimal_route_quality = 0
optimal_route_quality_simulated = optimal_route_assessment(array,array_geometry,bee_data,initial_probability_matrix,silent_sim=False,optimal_route_quality)
"""