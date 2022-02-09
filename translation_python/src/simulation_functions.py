'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
import copy
import itertools


import bout_functions
import management_of_data_functions
import geometry_functions
import environment_generation_functions
import learning_functions
import optimal_route_assessment_functions

current_working_directory = os.getcwd()

# Main  --------------------------------------------------------------

def make_arrays_and_output_folders(silent_sim) : 
  try : 
    os.mkdir("Output")
    if not silent_sim : 
      print("Output folder created")
  except :
    if not silent_sim : 
      print("Output folder already existing.")

  try : 
    os.mkdir("Arrays")
    if not silent_sim : 
      print("Arrays folder created")
  except :
    if not silent_sim : 
      print("Arrays folder already existing.")

def create_name_of_test(environment_type,test_name_general,number_of_parameter_sets,silent_sim) :
    time_now = (datetime.now()).strftime("%Y%m%d%Y%H%M%S")
    test_name = environment_type+"-"+test_name_general+"-param_set"+str(number_of_parameter_sets)+"-"+time_now
    if not silent_sim : 
      print("Starting simulation for test : "+test_name)
    return(test_name)


def simulation(test_name_general,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,dist_factor,number_of_bouts,number_of_simulations,silent_sim,sensitivity_analysis=False):
  """
  Description:
  Inputs:
    parameters_loop: dictionary key = name of individual parameters and parameters_loop[key] = list of different parameters to test. parameters_loop[key][i] is either a single value or a liste of values for each individual
    Careful: parameters_loop["use_Q_table"][i] and parameters_loop["initialize_Q_table"][i] is always a single value
  Ouputs:
  """

  # Create Output directory in the current working directory.
  make_arrays_and_output_folders(silent_sim)


  # If environment_type is not a "generate", there is no need for multiple arrays.
  if(array_info["environment_type"]!="generate") : 
    number_of_arrays = 1;
  print("number_of_arrays: ", number_of_arrays)

  # Get starting time of simulation to get computation time.
  start_of_simulation = time.time()

  # Successive loops for each parameter. Thus, all parameter combinations are tested. This code allows to add new parameters in parameters_loop without changin the loop code
  list_of_names_of_parameters = []
  for param in parameters_loop.keys() : 
    list_of_names_of_parameters.append(param)
  print("number of param: ", len(list_of_names_of_parameters))

  number_of_parameter_sets = 0 # used to create the name of the test

  for parameter_values in itertools.product(*[parameters_loop[param] for param in list_of_names_of_parameters]) : 

    parameter_values = list(parameter_values)
    print("parameter values:",parameter_values)

    # Initializing -------------------------------------------------------------------------------------------------
    number_of_parameter_sets += 1

    # Retrieve some general parameters

    index_use_Q_learning = list_of_names_of_parameters.index("use_Q_learning")
    use_Q_learning = parameter_values[index_use_Q_learning]

    index_initialize_Q_table = list_of_names_of_parameters.index("initialize_Q_table")
    initialize_Q_table = parameter_values[index_initialize_Q_table]

    print('initialize_Q_table:', initialize_Q_table)
    # Create test name according to parameter values

    test_name = create_name_of_test(array_info["environment_type"],test_name_general,number_of_parameter_sets,silent_sim)
    # Create the output folder for this test in the Output directory

    output_folder_of_test = current_working_directory + "\\Output\\"+ test_name
    os.mkdir(output_folder_of_test)

    # Complete list of individual parameters. These are initialized with parameters_loop

    param_indiv = dict(zip(list_of_names_of_parameters,parameter_values))

    # Simulation ---------------------------------------------------------------------------------------------------

    # Parameters tracked during the simulation for each bees
    param_tracking = {"number_of_resources_foraged": 0, "probability_of_winning" : 1/number_of_bees, "bout_finished" : False, "distance_travelled":0.}
    # Initialize the parameter dataframe for each bee
    bee_data = management_of_data_functions.initialize_bee_data(number_of_bees,param_tracking,param_indiv)
    print("bee data initialized: ")

    for array_number in range (number_of_arrays): 

      # Generate array
      array_geometry, array_info, array_folder = environment_generation_functions.create_environment(array_info, array_number, reuse_generated_arrays, silent_sim)
      print("array_geometry:")
      print(array_geometry)

      # Initialize the list of probability matrices (always useful to assess optimal route quality)
      if not silent_sim :
        print("Initializing probability matrices.")
      initial_probability_matrix_list = geometry_functions.initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees)

      # Modify these matrices depending on options
      for ind in range (number_of_bees) :
        if not bee_data["allow_nest_return"][ind] : 
          initial_probability_matrix_list[ind][:,0] = 0
          initial_probability_matrix_list[ind] = geometry_functions.normalize_matrix_by_row(initial_probability_matrix_list[ind])

      # Save a copy of the initial probability matrix
      path = array_folder + "\\probability_matrix.csv"
      print("saving proba matrix at: ", path)
      np.savetxt(path, initial_probability_matrix_list[0], delimiter=',')

      # Initialize the list of Q-tables (maybe later will be used to assess optimal route quality)
      if not silent_sim :
        print("Initializing Q tables.")

      initial_Q_table_list = learning_functions.initialize_Q_table_list (initialize_Q_table, array_geometry, dist_factor, number_of_bees)

      # Save a copy of the initial Q table 
      path = array_folder + "\\Q_table.csv"
      np.savetxt(path, initial_Q_table_list[0], delimiter=',')

      # Get maximum route quality of the array (simulating _ 1Ind for 30 each bouts to try and find the optimal route).
      optimal_route_quality = optimal_route_assessment_functions.retrieve_optimal_route(array_info["array_ID"],array_geometry,bee_data,initial_probability_matrix_list,array_folder,silent_sim,0,number_of_bees=1)
      if not silent_sim : 
        print("Optimal route quality: "+str(optimal_route_quality))

      # Get maximum route quality for 2 ind of the array (simulating _ 2Ind for 30 each bouts to try and find the optimal route).
      optimal_route_quality_2_ind = optimal_route_assessment_functions.retrieve_optimal_route(array_info["array_ID"],array_geometry,bee_data,initial_probability_matrix_list,array_folder,silent_sim,0,number_of_bees=2)
      if not silent_sim : 
        print("Optimal route quality for 2 individuals: "+str(optimal_route_quality_2_ind))

      # Initialize distance matrix
      matrix_of_pairwise_distances = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)

      # Create an Array folder for the current array in the Output folder
      output_folder_of_sim = output_folder_of_test + "\\Array"+"{:02d}".format(array_number)
      os.mkdir(output_folder_of_sim)

      # Create a dataframe of information to be retrieve in further analyses (and remember what parameters were used).
      bee_info = management_of_data_functions.initialize_bee_info(number_of_bees,param_indiv,array_info["array_ID"])
      pd.DataFrame(bee_info).to_csv(path_or_buf = output_folder_of_sim+'\\bee_info.csv', index = False)

      # Save array info and array geometry in this folder
      array_info_saved = copy.deepcopy(array_info)
      for key in array_info_saved : 
        array_info_saved[key] = [array_info_saved[key]]
      pd.DataFrame(array_info_saved).to_csv(path_or_buf = output_folder_of_sim+'\\array_info.csv', index = False)
      array_geometry.to_csv(path_or_buf = output_folder_of_sim+'\\array_geometry.csv', index = False)

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
          for bee in range (number_of_bees) : 
            matrix_of_bee_data[i+bee,2]=bee
          matrix_of_bee_data[i:(i+number_of_bees),4] = bee_data["number_of_resources_foraged"]
          matrix_of_bee_data[i:(i+number_of_bees),5] = current_bout["route_quality"]

          list_of_visitation_sequences.append(current_bout["sequences"])
          i=i+number_of_bees

      print("matrix_of_bee_data")
      #print(matrix_of_bee_data)
      print("list of visitation seq")
      #print(list_of_visitation_sequences)

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

    # Video output  ---------------------------------------------------------------------------------------------------------
    # No yet developped 

    # End -------------------------------------------------------------------------------------------------------------------

    end_of_simulation = time.time()
    duration_of_simulation = end_of_simulation - start_of_simulation
    print("Simulation completed in "+str(round(duration_of_simulation,5))+" seconds.")