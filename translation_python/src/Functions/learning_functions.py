'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd
import geometry_functions

# Learning --------------------------------------------------------------

def apply_learning (route, probability_matrix_list,individual_flower_outcome,bee_data,bee_info,route_quality,ind) : 
    """
    Description:
      Change the probability matrix of a individual depending on its last performed bout.
    Inputs:
      route: vector of the index of visited flowers during a bout
      probability_matrix: matrix of array size * array size, with probabilities to do each vector linking two flowers
      flower_outcome_matrix : matrix of array size * array size, with values of -1 (negative outcome), 0 (no visit) or 1 (positive outcome).
      bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
      route_quality : numeric value of last route quality experienced. used in routeCompare situations.
    Outputs:
      probability matrix: updated probability matrix
    """
    probability_matrix = probability_matrix_list[ind]
    flower_outcome_matrix = individual_flower_outcome[ind]

    if np.shape(probability_matrix) != np.shape(flower_outcome_matrix) : 
      raise ValueError ("probability_matrix and flower_outcome matrices of different lengths!")

    # Apply learning and abandon factor to flower outcomes
    nrow,ncol = np.shape(probability_matrix)

    if not (bee_info["use_route_compare"][0]) : 
      # No route comparison, probe all values in the flower_outcome_matrix to input the changes in probabilities.
      factor_matrix = np.ones((nrow,ncol))

      for row in range (nrow): 
        for col in range (ncol) : 

          if flower_outcome_matrix[row,col] == 1 :
            factor_matrix[row,col] = bee_info["learning_factor"][ind]

          if flower_outcome_matrix[row,col] == -1 :
            factor_matrix[row,col] = bee_info["abandon_factor"][ind]
    else : 
      factor_matrix = np.ones((nrow,ncol))

      # Check if new route is better or equal
      if (route_quality >= bee_data[ind,3]) and (bee_data[ind,0]==bee_info["crop_capacity"][ind]) and (bee_data[ind,3]>=0) : 

        # Add all vectors of the route as a positive outcome
        number_of_flowers_in_route = len(route)

        for i in range (number_of_flowers_in_route-1) : 

          current_flower = route[i]
          next_flower = route[i+1]
          factor_matrix[current_flower,next_flower] = bee_info["learning_factor"][ind]

      # Indifferent to the route comparison, apply the abandon.
      for row in range (nrow): 
        for col in range (ncol) : 

          if flower_outcome_matrix[row,col] == -1 :
            factor_matrix[row,col] = bee_info["abandon_factor"][ind]

    # Multiply the factor and the probability matrices element-wise and normalize it.
    probability_matrix = probability_matrix*factor_matrix
    probability_matrix = geometry_functions.normalize_matrix_by_row(probability_matrix)

    return(probability_matrix)


def softmax(values_vector,beta) : 
    """
    Description:
      Return the probabilities of choosing options characterised by some values by using a softmax decision function
    Inputs:
      values_vector: numpy array of values
      beta: inverse temperature parameter (must be positive)
    Outputs:
      Vector of probabilities 
    """
    if len(values_vector) == 0 : 
      return([])
    else : 
      values_vector = np.array(values_vector)
      return(np.exp(beta*values_vector)/np.sum(np.exp(beta*values_vector)))


def apply_online_Q_learning(Q_table,state,action,reward,alpha_pos,alpha_neg,gamma) : 
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
  delta = reward+gamma*np.max(Q_table[action,:])-Q_table[state,action]
  if delta >= 0 :
    Q_table[state,action] = Q_table[state,action] + alpha_pos*delta
  else : 
    Q_table[state,action] = Q_table[state,action] + alpha_neg*delta
  return(Q_table)


def apply_online_learning(probability_matrix,state,action,reward,learning_factor,abandon_factor) : 
  """
  Description: 
    Applies online learning by T. Dubois: probability_matrix[state,action] = probability_matrix[state,action]*factor where factor=learning_factor when reward>0 and factor=abandon_factor otherwise
  Inputs:
    probability_matrx: probability of going from one flower to another
    state: previous flower
    action: current flower
    reward: 1 if positive outcome, -1 if negative outcome
    learning_factor: >=1 will increase the probabilty of doing an action
    abandon_factor: <=1 will decrease the porbability of doing an action
  Outputs: 
    Updated probability matrix
  """ 
  if reward >0 : 
    factor = learning_factor
  else :
    factor = abandon_factor
  probability_matrix[state,action] = probability_matrix[state,action]*factor
  probability_matrix = geometry_functions.normalize_matrix_by_row(probability_matrix)
  return(probability_matrix)


def online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array,state,action,reward,alpha_pos,alpha_neg,gamma,learning_factor,abandon_factor) : 
  """
  Description: 
    Arbitration between the two apply_online_(Q)learning fuctions
  Inputs:
    learning_array: either probability matrix or Q_table
    state: previous flower
    action: current flower
    reward: CAREFUL The reward is specified for Q learning 
    learning_factor: >=1 will increase the probabilty of doing an action
    abandon_factor: <=1 will decrease the porbability of doing an action
    alpha_pos(neg): learnig rate for positive(negative) outcomes 
    gamma: temportal discounting factor
  Outputs: 
    Updated learning array  
  """
  matrix_of_pairwise_distances = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)
  max_distance_between_flowers = np.max(matrix_of_pairwise_distances)

  if use_Q_learning :
    if cost_of_flying : 
      reward  = reward - matrix_of_pairwise_distances[state,action]/max_distance_between_flowers
    learning_array = apply_online_Q_learning(learning_array,state,action,reward,alpha_pos,alpha_neg,gamma)
  else : 
    learning_array = apply_online_learning(learning_array,state,action,reward,learning_factor,abandon_factor)

  return(learning_array)