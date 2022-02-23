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



def simulation_loop(initial_learning_array_list,number_of_simulations,number_of_bouts,number_of_bees,optimal_route_quality_1_ind,optimal_route_quality_2_ind,bee_info,array_geometry,silent_sim,array_folder,output_folder_of_sim,sensitivity_analysis,stochasticity):

  '''
  Description:
    Loop to make the simulation
  Inputs:
    number_of_simulations, number_of_bouts, number_of_bees: number of simulations, bouts and bees
    optimal_route_quality_1/2_ind: current approximation of the optimal route quality for this array for 1/2 bees
    bee_info: pandas dataframe with important parameters of the bees
    array_geometry: dataframe of information about the geometry of the environment
    silent_sim: if True, prevents from printing
    array_folder: path of the /Array folder
    output_folder_of_sim: path of the Output/specific_simulation folder
    initial_learning_array_list : numpy array of size (number_of_bees, number_of_flowers,number_of_flowers) (number_of_flowers icludes nest) giving for each bee its learning array at the beginnig of the simulation
    sensitivity_analysis: if True, performs sensitivitya analysis
    stochasticity: if False, deactivate the stochasticity in the bout_function
  Outputs:
    updated optimal_route_quality_1/2_ind
  '''

  saved_optimal_route_quality_1_ind = optimal_route_quality_1_ind # To compare with the optimal route quality after the simulation and update the corresponding .csv file
  saved_optimal_route_quality_2_ind = optimal_route_quality_2_ind # To compare with the optimal route quality after the simulation and update the corresponding .csv file

  # Initialize output structures
  list_of_visitation_sequences = [] # Need to store the bee routes in a dynamic structure and not a numpy array because we cannot know in advance the lzngth of the longest sequence 
  matrix_of_route_qualities = np.full((number_of_simulations*number_of_bouts*number_of_bees,6),None) # sim, bout, bee, distance_with_previous_learning_array, number_of_resources_foraged, route_quality

  i = 0 # index used to fill up the matrix_of_route_qualities

  # Sim loop
  for sim in range (number_of_simulations): 

    # Initialize simulation objects
    bee_sequences = []

    learning_array_list = copy.deepcopy(initial_learning_array_list)

    # Bout loop
    for bout in range (number_of_bouts) :

      # Save learning array list before bout for sensitivity analysis
      if sensitivity_analysis : 
        previous_learning_array_list = copy.deepcopy(learning_array_list)

      bee_route,route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_resources_foraged = bout_functions.simulate_bout(bout,bee_info,learning_array_list,array_geometry,optimal_route_quality_1_ind,optimal_route_quality_2_ind,stochasticity)
      
      # For Sensitivity Analysis, compare previous_learning_array_list and learning_array_list
      if sensitivity_analysis : 
        for bee in range (number_of_bees):
          previous_matrix = previous_learning_array_list[bee]
          next_matrix = learning_array_list
          difference = np.sum(np.abs(previous_matrix-next_matrix))
          matrix_of_route_qualities[i+bee,3]=difference

      # Fill up the matrix of route qualities
      matrix_of_route_qualities[i:(i+number_of_bees),0] = sim
      matrix_of_route_qualities[i:(i+number_of_bees),1] = bout
      for bee in range (number_of_bees) : matrix_of_route_qualities[i+bee,2]=bee
      matrix_of_route_qualities[i:(i+number_of_bees),4] = number_of_resources_foraged
      matrix_of_route_qualities[i:(i+number_of_bees),5] = route_qualities

      # Add the routes to the list of visitation sequences
      list_of_visitation_sequences.append(bee_route)

      i=i+number_of_bees

  # Updating the optimal route qualities

  if optimal_route_quality_1_ind > saved_optimal_route_quality_1_ind : 
    if not silent_sim:
      print("A better optimal route quality was found for 1 individual: ",optimal_route_quality_1_ind,". Previous route quality was: ", saved_optimal_route_quality_1_ind)
    pd.DataFrame({"optimal_route":[optimal_route_quality_1_ind]}).to_csv(path_or_buf = array_folder+'\\optimal_route_1_ind.csv', index = False)

  if optimal_route_quality_2_ind > saved_optimal_route_quality_2_ind : 
    if not silent_sim:
      print("A better optimal route quality was found for 2 individual: ",optimal_route_quality_2_ind,". Previous route quality was: ", saved_optimal_route_quality_2_ind)
    pd.DataFrame({"optimal_route":[optimal_route_quality_2_ind]}).to_csv(path_or_buf = array_folder+'\\optimal_route_2_ind.csv', index = False)


  # Formatting list of sequences into a matrix         
  maximum_length_of_a_route = 0 

  for visited_sequence in range (len(list_of_visitation_sequences)):
    maximum_length_of_a_route = max(maximum_length_of_a_route,len(list_of_visitation_sequences[visited_sequence][0]))

  matrix_of_visitation_sequences = np.full((number_of_simulations*number_of_bouts*number_of_bees,maximum_length_of_a_route+3),-1) # sim, bout, bee, sequence

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

  # Save the outputs
  np.savetxt(output_folder_of_sim+"\\matrix_of_visitation_sequences.csv",matrix_of_visitation_sequences, delimiter=',',fmt='%i')

  route_quality_dataframe = pd.DataFrame(matrix_of_route_qualities,columns=["simulation","bout","bee","distance_with_previous_learning_array","number_of_resources_foraged","absolute_quality"])
  route_quality_dataframe.to_csv(path_or_buf = output_folder_of_sim+'\\route_quality_DF.csv', index = False)

  if not silent_sim : 
    print("Simulations ended for the current test.\n")

  return(optimal_route_quality_1_ind,optimal_route_quality_2_ind)



def simulation(current_working_directory,experiment_name,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,number_of_bouts,number_of_simulations,silent_sim,stochasticity,sensitivity_analysis):
  """
  Description:
  Inputs:
    current_working_directory: current working directory (of the main.py file)
    experiment_name: identification for the experiment
    array_info: dictionary of information about the array generation 
    number_of_arrays: number of environements used
    parameters_loop: dictionary giving for each parameter the value of list of values that will be tested (see parameters.py for a full description of the architercture of the parameters).
    number_of_bees: number of bees foraging
    reuse_generated_arrays: if True, will reused parameter-matching generated arrays
    number_of_bouts: number of bouts per simulation
    number_of_simulations: number of simulations
    silent_sim: if True, prevents from printing
    sensitivity_analysis: if True, will compute the absolute difference between successive learning arrays for each bee
    stochasticity: if False, deactivate the stochasticity in the bout_function
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

    number_of_parameter_sets +=1
    parameter_values = list(parameter_values)

    # Initializing -------------------------------------------------------------------------------------------------

    output_folder_of_test, bee_info = management_of_data_functions.initialize_data_of_current_test(list_of_names_of_parameters,parameter_values,array_info,experiment_name,number_of_parameter_sets,silent_sim,current_working_directory,number_of_bees)

    # Simulation ---------------------------------------------------------------------------------------------------

    for array_number in range (number_of_arrays): 

      array_geometry, array_info, array_folder, optimal_route_quality_1_ind, optimal_route_quality_2_ind, output_folder_of_sim,initial_learning_array_list = management_of_data_functions.initialize_data_of_current_array(array_info, array_number, reuse_generated_arrays, current_working_directory, silent_sim, number_of_bees,bee_info,output_folder_of_test)
      optimal_route_quality_1_ind,optimal_route_quality_2_ind = simulation_loop(initial_learning_array_list,number_of_simulations,number_of_bouts,number_of_bees,optimal_route_quality_1_ind, optimal_route_quality_2_ind,bee_info,array_geometry,silent_sim,array_folder,output_folder_of_sim,sensitivity_analysis,stochasticity)

    # Video output: not developed yet  ---------------------------------------------------------------------------------------------------------

    # End -------------------------------------------------------------------------------------------------------------------


  end_of_simulation = time.time()
  duration_of_simulation = end_of_simulation - start_of_simulation
  print("Simulation completed in "+str(round(duration_of_simulation,5))+" seconds.")