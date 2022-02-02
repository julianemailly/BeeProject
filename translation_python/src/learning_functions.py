'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import geometry_functions
import pandas as pd

# Learning --------------------------------------------------------------

def apply_learning (route, probability_matrix,flower_outcome_matrix,bee_data,route_quality) : 
    """
    Description:
      Change the probability matrix of a individual depending on its last performed bout.
    Inputs:
      route: vector of the index of visited flowers during a bout
      probability_matrix: matrix of array size * array size, with probabilities to do each vector linking two flowers
      flower_outcome_matrix : matrix of array size * array size, with values of -1 (negative outcome), 0 (no visit) or 1 (positive outcome).
      bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation
      route_quality : numeric value of last route quality experienced. used in routeCompare situations.
    Outputs:
      probability matrix: updated probability matrix
    """
    if np.shape(probability_matrix) != np.shape(flower_outcome_matrix) : 
      raise ValueError ("probability_matrix and flower_outcome matrices of different lengths!")

    # Apply learning and abandon factor to flower outcomes
    nrow,ncol = np.shape(probability_matrix)

    if not (bee_data["route_compare"][0]) : 

      # No route comparison, probe all values in the flower_outcome_matrix to input the changes in probabilities.
      factor_matrix = np.ones((nrow,ncol))

      for row in range (nrow): 
        for col in range (ncol) : 

          if flower_outcome_matrix[row,col] == 1 :
            factor_matrix[row,col] = bee_data["learning_factor"][0]

          if flower_outcome_matrix[row,col] == -1 :
            factor_matrix[row,col] = bee_data["abandon_factor"][0]
    else : 
      factor_matrix = np.ones((nrow,ncol))

      # Check if new route is better or equal
      if (route_quality >= bee_data["best_route_quality"][0]) and (bee_data["number_of_foraged_resources"].equals(bee_data["crop_capacity"])) and (bee_data["best_route_quality"][0]>0) : ## /!\ Assuming crop capacity of 5.

        # Add all vectors of the route as a positive outcome
        number_of_flowers_in_route = len(route)

        for i in range (number_of_flowers_in_route-1) : 

          current_flower = route[i]
          next_flower = route[i+1]
          factor_matrix[current_flower,next_flower] = bee_data["learning_factor"][0]

      # Indifferent to the route comparison, apply the abandon.
      for row in range (nrow): 
        for col in range (ncol) : 

          if flower_outcome_matrix[row,col] == -1 :
            factor_matrix[row,col] = bee_data["abandon_factor"][0]

    # Multiply the factor and the probability matrices element-wise and normalize it.
    probability_matrix = probability_matrix*factor_matrix
    probability_matrix = geometry_functions.normalize_matrix_by_row(probability_matrix)

    return(probability_matrix)


def softmax(values_vector,beta_QL) : 
    """
    Description:
      Return the probabilities of choosing options characterised by some values by using a softmax decision function
    Inputs:
      values_vector: numpy array of values
      beta_QL: inverse temperature parameter (must be positive)
    Outputs:
      Vector of probabilities 
    """
    values_vector = np.array(values_vector)
    return(np.exp(beta_QL*values_vector)/np.sum(np.exp(beta_QL*values_vector)))


def apply_online_Q_learning(Q_table,state,action,reward,alpha_pos,alpha_neg,gamma_QL) : 
  """
  Description: 
    A bee with Q Table (QTable), is in state (state), does action (action), gets a reward (reward)
    Apply Q Learning algorithm: QTable[state,action]=QTable[state,action]+alpha*(reward+gamma*max(Q(nextState,b)/b in actions)-Q[state,action])
    With alpha: learning rate, gamma: temporal discount factor
    Here, action also corresponds to the next state since completely deterministic environment so max(Q(nextState,b)/b in actions)=max(Q(action,b)/b in actions)
    Two RL systems: one that reinforces positively some values (use alphaPos), one that reinforces negatively some values (use alphaNeg)
  Inputs:
    see description
  Outputs: 
    Updated Q table
  """
  delta = reward+gamma_QL*np.max(Q_table[action,:])-Q_table[state,action]
  if delta >= 0 :
    Q_table[state,action] = Q_table[state,action] + alpha_pos*delta
  else : 
    Q_table[state,action] = Q_table[state,action] + alpha_neg*delta
  return(Q_table)


def initialize_Q_table_list (initialize_Q_table, array_geometry, dist_factor, number_of_bees) : 
  """
  Description: 
    Initialize the Q table of each bee according to a preselected rule
  Inputs:
    initialize_Q_table: string, rule for initialization of Q matrix
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    dist_factor: float, parameter to estimate the probability of going from a flower to another
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation.
  Outputs: 
    List of Q tables
  """
  number_of_states = len(array_geometry.index)
  if initialize_Q_table == "distance" : 
    return(geometry_functions.initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees))
  elif initialize_Q_table == "zero" : 
    return([np.zeros((number_of_states,number_of_states)) for bee in range (number_of_bees)])
  elif initialize_Q_table == "noisydist" :
    Q_list = geometry_functions.initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees)
    for bee in range (number_of_bees) :
      Q_list[bee] = Q_list[bee] + 0.1*np.random.normal(loc=0,scale=1, size=(number_of_states,number_of_states))
    return(Q_list)