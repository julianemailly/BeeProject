'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy
import geometry_functions
import optimal_route_assessment_functions
import environment_generation_functions

# Management of data  --------------------------------------------------------------

def initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees,allow_nest_return_list) : 
  """
  Description: 
    Generates the probability matrix for a given array for each bee
  Inputs:
      array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
      dist_factor: float, parameter to estimate the probabilities 
      number_of_bees: integer giving the total number_of_bees
      allow_nest_return_list: list of allow_nest_return parameter for each bee
    Outputs: 
      initial_probability_matrix_list: list of probability matrices (one for each bee)
  """
  distance_between_flowers = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)
  initial_probability_matrix_list = [geometry_functions.normalize_matrix_by_row (geometry_functions.give_probability_of_vector_with_dist_factor(distance_between_flowers,dist_factor)) for bee in range (number_of_bees)]

  # Modify these matrices depending on options
  for ind in range (number_of_bees) :
    if not allow_nest_return_list[ind] : 
      initial_probability_matrix_list[ind][:,0] = 0
      initial_probability_matrix_list[ind] = geometry_functions.normalize_matrix_by_row(initial_probability_matrix_list[ind])

  return (np.array(initial_probability_matrix_list))

def initialize_Q_table_list (initialize_Q_table, array_geometry, dist_factor, number_of_bees,allow_nest_return_list) : 
  """
  Description: 
    Initialize the Q table of each bee according to a preselected rule
  Inputs:
    initialize_Q_table: string, rule for initialization of Q matrix
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    dist_factor: float, parameter to estimate the probability of going from a flower to another
    allow_nest_return_list: list of allow_nest_return parameter for each bee
  Outputs: 
    List of Q tables
  """
  number_of_states = len(array_geometry.index)

  if initialize_Q_table == "distance" : 
    return(initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees,allow_nest_return_list))

  elif initialize_Q_table == "zero" : 
    return([np.zeros((number_of_states,number_of_states)) for bee in range (number_of_bees)])

  elif initialize_Q_table == "noisydist" :
    Q_list = initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees,allow_nest_return_list)

    for bee in range (number_of_bees) :
      Q_list[bee] = Q_list[bee] + 0.1*np.random.normal(loc=0,scale=1, size=(number_of_states,number_of_states))
      
    return(np.array(Q_list))



def initialize_bee_info(number_of_bees,parameters_dict) :
  """
  Description:
    Sets up a data frame of parameters of bees.
  Inputs:
    number_of_bees: integer, number of bees
    parameters_dict: dictionnary with parameters of the bees
    array_ID: string with the ID of the array
  Outputs:
    bee_info: pandas dataframe with information about bees that will be stored at the end. nrows = number_of_bees, ncol = number of attributes of parameters_dict+2
  """
  for key in parameters_dict : 
    if not isinstance(parameters_dict[key],list) :
      parameters_dict[key] = [parameters_dict[key] for k in range (number_of_bees)] 
    else : 
      if len(parameters_dict[key]) != number_of_bees : 
        raise ValueError("Impossible to initialize dataframe of bee info because of parameter "+str(key)+" contains lists whose number of elements is different from the number of bees.\nPlease refer to parameter.py for a full description of the initialization of the parameters")  
  dict_of_bee_info = {"ID": [bee for bee in range (number_of_bees)]}
  dict_of_bee_info.update(parameters_dict)
  bee_info = pd.DataFrame(dict_of_bee_info)
  return(bee_info)

def add_array_ID_to_bee_info(bee_info, array_ID,number_of_bees) :
  bee_info["array_ID"] = [array_ID for k in range (number_of_bees)]
  return()


def make_arrays_and_output_folders(silent_sim) : 
  """
  Description:
    Creates 'Arrays' and 'Output' folders in the /src directory
  Inputs:
    silent_sim: if True, prevents from printing
  Outputs:
    None
  """
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


def get_value_of_parameter_in_current_test(name_of_parameter,list_of_names_of_parameters,parameter_values): 
  """
  Description:
    Get the value of a parameter
  Inputs:
    name_of_parameter: string, name of the parameter as referenced in list_of_names_of_parameters
    list_of_names_of_parameters: list of names of parameters
    parameter_values: list of values of parameters in list_of_names_of_parameters
  Outputs:
    value_of_parameter: value of the paameter
  """
  index_of_parameter = list_of_names_of_parameters.index(name_of_parameter)
  value_of_parameter = parameter_values[index_of_parameter] 
  return(value_of_parameter)


def create_name_of_test(environment_type,experiment_name,number_of_parameter_sets,silent_sim) :
  """
  Description:
    Creates name of the test
  Inputs:
    environment_type: name of the array that will be used or "generate" if the array is to be generated procedurally
    experiment_name: identification for the experiment
    number_of_parameter_sets: number of sets of parameters already done 
    silent_sim: if True, prevents from printing
  Outputs:
    test_name: name of the test
  """
  time_now = (datetime.now()).strftime("%Y%m%d%Y%H%M%S")
  test_name = environment_type+"-"+experiment_name+"-param_set"+str(number_of_parameter_sets)+"-"+time_now
  if not silent_sim : 
    print("\nStarting simulation for test : "+test_name)
  return(test_name)


def initialize_data_of_current_test(list_of_names_of_parameters,parameter_values,array_info,experiment_name,number_of_parameter_sets,silent_sim,current_working_directory,number_of_bees):
  """
  Description:
    Initializes the data of the current test
  Inputs:
    list_of_names_of_parameters: list of names of parameters
    parameter_values: list of values of parameters in list_of_names_of_parameters
    array_info: dictionary of information about the environment
    environment_type: name of the array that will be used or "generate" if the array is to be generated procedurally
    experiment_name: identification for the experiment
    number_of_parameter_sets: number of sets of parameters already done 
    silent_sim: if True, prevents from printing
    current_working_directory: name of current working directory
    number_of_bees: number of bees
  Outputs:
    Some useful variables for the test
  """

  # Create test name according to parameter values
  test_name = create_name_of_test(array_info["environment_type"],experiment_name,number_of_parameter_sets,silent_sim)

  # Create the output folder for this test in the Output directory
  output_folder_of_test = current_working_directory + "\\Output\\"+ test_name
  os.mkdir(output_folder_of_test)

  # Complete list of individual parameters. These are initialized with parameters_loop
  parameters_dict = dict(zip(list_of_names_of_parameters,parameter_values))

  # Create a dataframe of information to be retrieve in further analyses and passed to the simulation functions (and remember what parameters were used).
  bee_info = initialize_bee_info(number_of_bees,parameters_dict)

  return( test_name, output_folder_of_test, parameters_dict,bee_info)




def initialize_data_of_current_array(array_info, array_number, reuse_generated_arrays, current_working_directory, silent_sim, number_of_bees,bee_info,parameters_dict,output_folder_of_test):
  # Generate array
  array_geometry, array_info, array_folder = environment_generation_functions.create_environment(array_info, array_number, reuse_generated_arrays, current_working_directory, silent_sim)

  # Initialize learning array list
  use_Q_learning = bee_info["use_Q_learning"][0]
  
  if use_Q_learning : 
    initial_learning_array_list = initialize_Q_table_list (bee_info["initialize_Q_table"][0], array_geometry, bee_info["dist_factor"][0], number_of_bees,bee_info["allow_nest_return"])
  else : 
    initial_learning_array_list = initialize_probability_matrix_list(array_geometry, bee_info["dist_factor"][0], number_of_bees,bee_info["allow_nest_return"])


  # Add array_ID to bee_info
  add_array_ID_to_bee_info(bee_info, array_info["array_ID"],number_of_bees)

  # Get maximum route quality of the array (simulating _ 1Ind for 30 each bouts to try and find the optimal route).
  optimal_route_quality_1_ind = optimal_route_assessment_functions.retrieve_optimal_route(array_geometry,bee_info,array_folder,silent_sim,0,None,number_of_bees=1)
  if not silent_sim : 
    print("Optimal route quality for 1 individual: "+str(optimal_route_quality_1_ind))

  # Get maximum route quality for 2 ind of the array (simulating _ 2Ind for 30 each bouts to try and find the optimal route).
  optimal_route_quality_2_ind = optimal_route_assessment_functions.retrieve_optimal_route(array_geometry,bee_info,array_folder,silent_sim,0,0,number_of_bees=2)
  if not silent_sim : 
    print("Optimal route quality for 2 individuals: "+str(optimal_route_quality_2_ind))

  # Create an Array folder for the current array in the Output folder
  output_folder_of_sim = output_folder_of_test + "\\Array"+"{:02d}".format(array_number)
  os.mkdir(output_folder_of_sim)

  # Save bee_info
  pd.DataFrame(bee_info).to_csv(path_or_buf = output_folder_of_sim+'\\bee_info.csv', index = False)

  return(array_geometry, array_info, array_folder, optimal_route_quality_1_ind, optimal_route_quality_2_ind, output_folder_of_sim,initial_learning_array_list)
