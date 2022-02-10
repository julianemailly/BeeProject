'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd
import time
import copy
import itertools


import bout_functions
import management_of_data_functions


# Main  --------------------------------------------------------------



def get_list_of_parameters_names(parameters_loop):
  """
  Description:
    Get the keys of a dictionary and store them in a list
  Inputs:
    parameters_loop: disctionary of parameters
  Outputs:
    list_of_names_of_parameters: list of names of parameters
  """
  list_of_names_of_parameters = []
  for param in parameters_loop.keys() : 
    list_of_names_of_parameters.append(param)
  return(list_of_names_of_parameters)



def simulation_loop(number_of_simulations,number_of_bouts,number_of_bees,optimal_route_quality,bee_data,array_geometry,silent_sim,array_folder,output_folder_of_sim,use_Q_learning,initial_Q_table_list,initial_probability_matrix_list,sensitivity_analysis,matrix_of_pairwise_distances):

  '''
  Description:
    Loop to make the simulation
  Inputs:
    number_of_simulations, number_of_bouts, number_of_bees: number of simulations, bouts and bees
    optimal_route_quality: current approximation of the optimal route quality for this array
    bee_data: data about bees to be updated
    array_geometry: dataframe of information about the geometry of the environment
    silent_sim: if True, prevents from printing
    array_folder: path of the /Array folder
    output_folder_of_sim: path of the Output/specific_simulation folder
    use_Q_learning: if True, use the Q-Learning algorithm instead of T. Dubois's
    initial_Q_table_list,initial_probability_matrix_list,
    sensitivity_analysis: if True, performs sensitivitya analysis
    matrix_of_pairwise_distances: matrix of distances between flowers
  Outputs:
    updated optimal_route_quality
  '''
  # Initialize output structures
  list_of_visitation_sequences = []
  matrix_of_bee_data = np.full((number_of_simulations*number_of_bouts*number_of_bees,6),None) # sim, bout, bee, distance_with_previous_learning_array, number_of_resources_foraged, route_quality

  i=0

  # Sim loop
  for sim in range (number_of_simulations): 

    # Initialize simulation objects
    bee_sequences = []
    if not use_Q_learning : 
      learning_array_list = initial_Q_table_list
    else : 
      learning_array_list = initial_probability_matrix_list

    # Bout loop
    for bout in range (number_of_bouts) :

      # Save learning array list before bout for sensitivity analysis
      if sensitivity_analysis : 
        previous_learning_array_list = copy.deepcopy(learning_array_list)

      management_of_data_functions.reboot_bee_data(bee_data)
      current_bout = bout_functions.simulate_bout(bout,array_geometry,learning_array_list,bee_data,optimal_route_quality,silent_sim,array_folder)
      
      # For Sensitivity Analysis, compare previous_learning_array_list and learning_array_list
      if sensitivity_analysis : 
        for bee in range (number_of_bees):
          previous_matrix = previous_learning_array_list[bee]
          next_matrix = learning_array_list
          difference = np.sum(np.abs(previous_matrix-next_matrix))
          matrix_of_bee_data[i+bee,3]=difference

      # Update variables: learning_array_list and bee_data are modified in place 

      optimal_route_quality = current_bout["optimal_route_quality"]

      matrix_of_bee_data[i:(i+number_of_bees),0] = sim
      matrix_of_bee_data[i:(i+number_of_bees),1] = bout
      for bee in range (number_of_bees) : matrix_of_bee_data[i+bee,2]=bee
      matrix_of_bee_data[i:(i+number_of_bees),4] = bee_data["number_of_resources_foraged"]
      matrix_of_bee_data[i:(i+number_of_bees),5] = current_bout["route_quality"]

      list_of_visitation_sequences.append(current_bout["sequences"])
      i=i+number_of_bees

  # Formatting raw data ---------------------------------------------------------------------------------------------------          
  max_length = 0

  for visit_seq in range (len(list_of_visitation_sequences)):
    max_length = max(max_length,len(list_of_visitation_sequences[visit_seq][0]))


  matrix_of_visitation_sequences = np.full((number_of_simulations*number_of_bouts*number_of_bees,max_length+3),0) # sim, bout, bee, sequence

  for sim in range (number_of_simulations):
    for bout in range (number_of_bouts) :
      sequences =  list_of_visitation_sequences[sim*number_of_bouts+bout]
      sequences_length = len(sequences[0])
      for bee in range (number_of_bees) :
        index_in_matrix =  sim*number_of_bouts*number_of_bees+bout*number_of_bees+bee
        matrix_of_visitation_sequences[index_in_matrix,0] = sim
        matrix_of_visitation_sequences[index_in_matrix,1] = bout
        matrix_of_visitation_sequences[index_in_matrix,2] = bee
        matrix_of_visitation_sequences[index_in_matrix,3:3+sequences_length] = sequences[bee]

  np.savetxt(output_folder_of_sim+"\\matrix_of_visitation_sequences.csv",matrix_of_visitation_sequences, delimiter=',',fmt='%i')

  # Compute absolute route quality
  absolute_route_quality =np.reshape(matrix_of_bee_data[:,-1],(number_of_simulations*number_of_bouts*number_of_bees,1))
  if optimal_route_quality != 0 :
    absolute_route_quality = absolute_route_quality/optimal_route_quality
  matrix_of_bee_data = np.concatenate((matrix_of_bee_data,absolute_route_quality),axis=1)

  route_quality_dataframe = pd.DataFrame(matrix_of_bee_data,columns=["simulation","bout","bee","distance_with_previous_learning_array","number_of_resources_foraged","relative_quality","absolute_quality"])
  
  route_quality_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\route_quality_DF.csv', index = False)

  # Save matrix of distances
  np.savetxt(output_folder_of_sim+'\\matrix_of_distances.csv',matrix_of_pairwise_distances,delimiter=',')

  if not silent_sim : 
    print("Simulations ended for the current test.\n")

  return(optimal_route_quality)



def simulation(current_working_directory,experiment_name,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,dist_factor,number_of_bouts,number_of_simulations,silent_sim,sensitivity_analysis=False):
  """
  Description:
  Inputs:
    experiment_name: identification for the experiment
    array_info: dictionary of information about the array generation 
    number_of_arrays: number of environements used
    parameters_loop: dictionary giving for each parameter the value of list of values that will be tested (see parameters.py for a full description of the architercture of the parameters).
    number_of_bees: number of bees foraging
    reuse_generated_arrays: if True, will reused parameter-matching generated arrays
    dist_factor: used when computing the initial learning array 
    number_of_bouts: number of bouts per simulation
    number_of_simulations: number of simulations
    silent_sim: if True, prevents from printing
    sensitivity_analysis: if True, will compute the absolute difference between successive learning arrays for each bee
  Ouputs:
    Stored in /Output folder
  """

  # Create Output directory in the current working directory.
  management_of_data_functions.make_arrays_and_output_folders(silent_sim)


  # If environment_type is not a "generate", there is no need for multiple arrays.
  if(array_info["environment_type"]!="generate") : 
    number_of_arrays = 1;

  # Get starting time of simulation to get computation time.
  start_of_simulation = time.time()

  # Successive loops for each parameter. Thus, all parameter combinations are tested. This code allows to add new parameters in parameters_loop without changin the loop code
  list_of_names_of_parameters = get_list_of_parameters_names(parameters_loop)

  number_of_parameter_sets = 0 # used to create the name of the test

  for parameter_values in itertools.product(*[parameters_loop[param] for param in list_of_names_of_parameters]) : 

    parameter_values = list(parameter_values)

    # Initializing -------------------------------------------------------------------------------------------------

    number_of_parameter_sets, use_Q_learning, initialize_Q_table, test_name, output_folder_of_test, param_indiv, param_tracking, bee_data = management_of_data_functions.initialize_data_of_current_test(list_of_names_of_parameters,parameter_values,array_info,experiment_name,number_of_parameter_sets,silent_sim,current_working_directory,number_of_bees)

    # Simulation ---------------------------------------------------------------------------------------------------

    for array_number in range (number_of_arrays): 

      array_geometry, array_info, array_folder, initial_probability_matrix_list, initial_Q_table_list, optimal_route_quality, optimal_route_quality_2_ind, matrix_of_pairwise_distances, output_folder_of_sim, bee_info = management_of_data_functions.initialize_data_of_current_array(array_info, array_number, reuse_generated_arrays, current_working_directory, silent_sim, dist_factor, number_of_bees, bee_data,initialize_Q_table,param_indiv,output_folder_of_test)

      optimal_route_quality = simulation_loop(number_of_simulations,number_of_bouts,number_of_bees,optimal_route_quality,bee_data,array_geometry,silent_sim,array_folder,output_folder_of_sim,use_Q_learning,initial_Q_table_list,initial_probability_matrix_list,sensitivity_analysis,matrix_of_pairwise_distances)


    # Video output: not developed yet  ---------------------------------------------------------------------------------------------------------

    # End -------------------------------------------------------------------------------------------------------------------

  end_of_simulation = time.time()
  duration_of_simulation = end_of_simulation - start_of_simulation
  print("Simulation completed in "+str(round(duration_of_simulation,5))+" seconds.")