'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd
import geometry_functions

# Learning --------------------------------------------------------------

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


def apply_online_Q_learning(bee,Q_table_list,state,action,reward,alpha_pos,alpha_neg,gamma) : 
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
  delta = reward+gamma*np.max(Q_table_list[bee,action,:])-Q_table_list[bee,state,action]
  if delta >= 0 :
    Q_table_list[bee,state,action] += alpha_pos*delta
  else : 
    Q_table_list[bee,state,action] += alpha_neg*delta


def apply_online_multiplicative_learning(bee,probability_matrix_list,state,action,reward,learning_factor,abandon_factor) : 
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
  probability_matrix_list[bee,state,action] *= factor
  sum_of_probabilities = np.sum(probability_matrix_list[bee,state,:])
  if sum_of_probabilities != 0 :
    probability_matrix_list[bee,state,:] /= sum_of_probabilities


def apply_online_learning(bee,cost_of_flying,array_geometry,use_Q_learning,learning_array_list,state,action,reward,alpha_pos,alpha_neg,gamma,learning_factor,abandon_factor) : 
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
  if reward is not None : 

    if use_Q_learning :
      if cost_of_flying : 
        matrix_of_pairwise_distances = geometry_functions.get_matrix_of_distances_between_flowers(array_geometry)
        max_distance_between_flowers = np.max(matrix_of_pairwise_distances)
        reward  = reward - matrix_of_pairwise_distances[state,action]/max_distance_between_flowers
      apply_online_Q_learning(bee,learning_array_list,state,action,reward,alpha_pos,alpha_neg,gamma)
    else : 
      apply_online_multiplicative_learning(bee,learning_array_list,state,action,reward,learning_factor,abandon_factor)