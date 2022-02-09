'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd


# Management of data  --------------------------------------------------------------

def initialize_bee_data(number_of_bees,param_tracking,param_indiv) : 
  """
  Description: 
    Sets up a data frame including all important informations on the foraging bees.
  Inputs: 
    number_of_bees: integer, number of bees
    param_tracking: dictionnary with parameters that must be tracked 
    param_indiv: dictionnary with parameters of the bees
  Outputs:
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation. nrows = number_of_bees, ncol = number of attributes of param_trancking+number of attributes of param_indiv+2
  """
  for key in param_indiv : 
    if not isinstance(param_indiv[key],list) :
      param_indiv[key] = [param_indiv[key] for k in range (number_of_bees)] 
    else : 
      if len(param_indiv[key]) != number_of_bees : 
        raise ValueError("Impossible to initialize dataframe of bee data because of parameter "+str(key)+" contains lists whose number of elements is different from the number of bees.")
  for key in param_tracking : 
    param_tracking[key] = [param_tracking[key] for k in range (number_of_bees)]  
  dict_of_bee_data = {"ID": [bee for bee in range (number_of_bees)]}
  dict_of_bee_data.update(param_tracking)
  dict_of_bee_data.update(param_indiv)
  bee_data = pd.DataFrame(dict_of_bee_data)
  return(bee_data)


def initialize_bee_info(number_of_bees,param_indiv,array_ID) :
  """
  Description:
    Sets up a data frame of parameters of bees.
  Inputs:
    number_of_bees: integer, number of bees
    param_indiv: dictionnary with parameters of the bees
    array_ID: string with the ID of the array
  Outputs:
    bee_info: pandas dataframe with information about bees that will be stored at the end. nrows = number_of_bees, ncol = number of attributes of param_indiv+2
  """
  for key in param_indiv : 
    if not isinstance(param_indiv[key],list) :
      param_indiv[key] = [param_indiv[key] for k in range (number_of_bees)] 
    else : 
      if len(param_indiv[key]) != number_of_bees : 
        raise ValueError("Impossible to inistialize dataframe of bee info because of parameter "+str(key)+" contains lists whose number of elements is different from the number of bees.")  
  dict_of_bee_info = {"ID": [bee for bee in range (number_of_bees)]}
  dict_of_bee_info.update(param_indiv)
  #dict_of_bee_info.update({'array_ID':array_ID})
  bee_info = pd.DataFrame(dict_of_bee_info)
  try : 
    bee_info.drop(columns = ["best_route_quality"])
  finally : 
    return(bee_info)


def reboot_bee_data(bee_data) : 
  """
  Description:
    Resets the parameters of beeData between bouts.
  Inputs:
    bee_data: pandas datafram containing relevant information about the bees
  Outputs:
    The updated bee_data dataframe
  """
  number_of_bees = len(bee_data["ID"])
  for ind in range (number_of_bees) : 

    bee_data.loc[ind,"number_of_resources_foraged"] = 0
    bee_data.loc[ind,"bout_finished"] = False
    bee_data.loc[ind,"distance_travelled"] = 0.