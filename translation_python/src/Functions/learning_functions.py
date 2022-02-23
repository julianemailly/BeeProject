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
    A bee with Q Table, is in state (state), does action (action), gets a reward (reward)
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
    bee: index of bee
    probability_matrx_list: list of probability matrices of going from one flower to another (one for each bee)
    state: previous flower
    action: current flower
    reward: >0 for positive RL, <=0 for negative RL
    learning_factor: >=1 will increase the probabilty of doing an action
    abandon_factor: 0<= and <=1, will decrease the porbability of doing an action
  Outputs: 
    Updated probability matrix list (in place)
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
    Arbitration between the two apply_online_q/muktiplicative_learning functions
  Inputs:
    bee: index of bee
    learning_array_list: either probability matrix list or Q_table list
    use_Q_learning: if True, will use Q-learning, else, will use T.Dubois' model
    cost_of_flying: if True and if use Q-Learning, the distance between two flowers will be integrated as  punishment in the rewad computation
    array_geometry: pandas dataframe giving the position of the different flowers
    state: previous flower
    action: current flower
    reward: CAREFUL The reward is specified for Q learning 
    learning_factor: >=1 will increase the probabilty of doing an action
    abandon_factor: <=1 will decrease the porbability of doing an action
    alpha_pos(neg): learnig rate for positive(negative) outcomes 
    gamma: temporal discounting factor
  Outputs: 
    Updated learning array list (in place)
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