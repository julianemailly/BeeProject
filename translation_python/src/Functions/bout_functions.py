'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import learning_functions
import geometry_functions

# Simulation functions  --------------------------------------------------------------

def give_omitted_destinations(ind,current_pos,previous_pos,array_of_vector_used,bee_info) :
  """
  Description:
    Returns the destinations that must be omitted when sampling the next destination
  Inputs:
    ind: index of the individual
    current_pos: curent position of the individual
    previous_pos: previous position of the individual
    array_of_vector_used: array giving the number of times a transition was used by a bee since the beginning of the bout (size number_of_flowers*number_of_flowers*number_of_bee)
    bee_info: pandas dataframe of the parameters 
  Outputs:
    omitted_destinations: vector of destinations to be omitted
  """ 
  omitted_destinations = [current_pos] # At least the self-loop is forbidden.

  if not (bee_info["allow_nest_return"][ind]) : # If condition, omit the nest.
    omitted_destinations.append(0)

  if bee_info["forbid_reverse_vector"][ind] and previous_pos is not None : # If condition, omit the previous visit.
    omitted_destinations.append(previous_pos)

  if bee_info["leave_after_max_fail"][ind] : 
    omitted_destinations=np.concatenate((omitted_destinations,np.where(array_of_vector_used[current_pos,:,ind]==bee_info["number_of_max_fails"][ind])[0]))

  omitted_destinations = np.unique(omitted_destinations) # Make sure each flower is mentioned only once.

  return(omitted_destinations)


def get_probabilities_of_choosing_flowers(use_Q_learning,learning_array_list,ind,current_pos,destination_is_available,use_dynamic_beta,beta_vector,bout,beta) :
  """
  Description:
    Give the probability of choosing among availabl destinations
  Inputs:
    use_Q_learning: if True will use a softmax function to give probabilities
    learning_array_list: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
    ind: index of individual
    current_pos: current position of individual
    destination_is_available: bool list such as destination_is_available[flower_ID] says if flower with ID flower_ID is amng the possible destinations
    use_dynamic_beta: bool, if True will use a dynamic beta parameter whose values are specified in beta_vector
    beta_vector: specifies the values of beta in case of dynamic beta
    bout: index of bout
    beta: specifies inverse temperature parameter if use_Q_Learning
  Outputs:
    probabilities: a vector of probabilities
  """  
  if not use_Q_learning : 
    probabilities = learning_array_list[ind][current_pos,destination_is_available]
    sum_of_probabilities = np.sum(learning_array_list[ind][current_pos,destination_is_available])
    if sum_of_probabilities !=0 : 
      probabilities = probabilities/sum_of_probabilities
  else : 
    if use_dynamic_beta : 
      probabilities = learning_functions.softmax(learning_array_list[ind][current_pos,destination_is_available],beta_vector[bout])
    else : 
      probabilities = learning_functions.softmax(learning_array_list[ind][current_pos,destination_is_available],beta)
  return(probabilities)


def sampling_next_destination(number_of_bees,ind,bee_route,bee_data,bee_info,array_of_vector_used,number_of_flowers,learning_array_list,bout,still_foraging,array_geometry): 
  """
  Description:
    Samples the next destination for an individual and updates the corresponding data
  Inputs:
    ind: index of individual
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    bee_data: numpy array of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    array_of_vector_used: array giving the number of times a transition was used by a bee since the beginning of the bout (size number_of_flowers*number_of_flowers*number_of_bee)
    number_of_flowers: ottal number of flowers + nest
    learning_array_list: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
    bout: index of bout
    still_foraging: vactor of indices of individuals that are still foraging
  Outputs:
    Updated still_foraging, bee_route. bee_data is already modified in place
  """ 
  # Retrieve parameters of the simulation
  beta = bee_info["beta"][ind]
  beta_vector = bee_info["beta_vector"][ind]
  use_Q_learning = bee_info["use_Q_learning"][0]
  use_dynamic_beta = bee_info["use_dynamic_beta"][0]
  use_online_reinforcement = bee_info["use_online_reinforcement"][0]

  number_of_flowers = len(array_geometry.index)
  number_of_visits = np.shape(bee_route)[1]

  # Retrieve the bee's current position
  current_pos = bee_route[ind,number_of_visits-2]

  if number_of_visits>2 : 
    previous_pos = bee_route[ind,number_of_visits-3]
  else : 
    previous_pos = None

  # Mark all destinations which must be omitted
  omitted_destinations = give_omitted_destinations(ind,current_pos,previous_pos,array_of_vector_used,bee_info)

  # Retrieve all potential destinations
  potential_destinations = np.array([flower for flower in range (number_of_flowers)])
  for omitted_destination in omitted_destinations : 
    potential_destinations = np.delete(potential_destinations, np.where(potential_destinations==omitted_destination))
  destination_is_available = [not (flower in omitted_destinations) for flower in range (number_of_flowers)]


  # Define probabilities
  probabilities = get_probabilities_of_choosing_flowers(use_Q_learning,learning_array_list,ind,current_pos,destination_is_available,use_dynamic_beta,beta_vector,bout,beta)

  # If no positive probabilities
  if (np.sum(probabilities)==0) : 
    if current_pos == 0 : # The bee was already in the nest: triggering end of bout
      bee_data[ind, 1] = 1. # bout finished
      still_foraging = np.delete(still_foraging,np.where(still_foraging==ind))
      #next???
      next_pos = 0
    else : 
      next_pos = 0
      bee_route[ind,-1] = 0 # Go back to nest (not trigerring end of bout). Useless because it was already at 0 but anyway

  else : # There is a potential destination


    if len(potential_destinations)>1 :
      next_pos = np.random.choice(a=potential_destinations,p=probabilities)
      bee_route[ind,-1] = next_pos
    else : 
      next_pos = potential_destinations[0] # Not sure why it is useful though
      bee_route[ind,-1] = next_pos


  # Update distance travelled
  bee_data[ind, 2] += geometry_functions.distance(array_geometry.loc[current_pos,'x':'y'],array_geometry.loc[next_pos,'x':'y'])

  # Check if the bee chose the nest
  if(bee_info["allow_nest_return"][ind]) : 
    if (bee_route[ind,-1]==0) and (bee_data[ind, 2]>0) : 
      bee_data[ind, 1] = 1. # bout finished

  return(still_foraging,bee_route)


def get_probability_of_winning(individuals_in_competition,bee_info):
  """
  Description:
    Normalizes the probabilities of winning of competing individuals
  Inputs:
    individuals_in_competition: indices of competing individuals
    bee_info: pandas dataframe of the parameters 
  Outputs:
    A vector of probabilities 
  """ 
  not_normalized_prob = []
  for ind in individuals_in_competition : 
    not_normalized_prob.append(bee_info["probability_of_winning"][ind])
  normalized_prob = not_normalized_prob/np.sum(np.array(not_normalized_prob))
  return(normalized_prob)


def find_flowers_in_competition(flowers_visited_this_bout):
  """
  Description:
    Finds the duplicates in a list
  Inputs:
    flowers_visited_this_bout: list of flowers visited this bout with duplicates
  Outputs:
    flowers_in_competition: duplicates in flowers_visited_this_bout
  """ 
  unique_flowers, count_occurences = np.unique(flowers_visited_this_bout, return_counts=True)
  flowers_in_competition = unique_flowers[count_occurences>1]
  
  # We exclude potential detections of nest.
  flowers_in_competition = np.delete(flowers_in_competition,np.where(flowers_in_competition==0))
  return(flowers_in_competition)


def competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_info):
  """
  Description:
    Gives the winner and losers of a competing interaction
  Inputs:
    flower: index of flower
    flowers_visited_this_bout: list of flowers visited this bout
    bee_info: pandas dataframe of the parameters 
  Outputs:
    interaction_winner: ID of winner
    interaction_losers: list of ID of losers
  """ 
  # Which individuals are in competition
  individuals_in_competition = np.where (np.array(flowers_visited_this_bout) == flower)[0]
  probability_of_winning  = get_probability_of_winning(individuals_in_competition,bee_info)
  # Which wins and which loses
  interaction_winner = np.random.choice(a=individuals_in_competition,p=probability_of_winning)
  interaction_losers = np.delete(individuals_in_competition,np.where(individuals_in_competition==interaction_winner))
  return(interaction_winner,interaction_losers)


def foraging_and_online_learning_loop(number_of_bees,bee_route,bee_data,bee_info,array_of_vector_used,learning_array_list,individual_flower_outcome,array_geometry,resources_on_flowers,bout):
  """
  Description:
    Each bee will decide on wht flower it will go next and will be either rewarded or punished.
  Inputs:
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    array_of_vector_used: if bee_data["leave_after_max_fail"], will storing for each vector flower_a,flower_b and for each individual the number of times the transition between flower_a and flower_b led to a negative outcome
    learning_array_list: list of arrays on which the learning process will occur (either probability matrix or Q table)
    individual_flower_outcome: list of arrays containing the outcome of each transition for each individual
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    resources_on_flowers: list of resources on each flower
  Outputs:
    updated bee_route
    resources_on_flowers, learning_array_list, array_of_vector_used, individual_flower_outcome, bee_data are also modified in place
  """ 

  # Retrieve parameters of the simulation
  alpha_pos = bee_info["alpha_pos"]
  alpha_neg = bee_info["alpha_neg"]
  gamma = bee_info["gamma"]
  use_Q_learning = bee_info["use_Q_learning"][0]
  use_online_reinforcement = bee_info["use_online_reinforcement"][0]
  cost_of_flying = bee_info["cost_of_flying"][0]
  learning_factor = bee_info["learning_factor"]
  abandon_factor = bee_info["abandon_factor"]

  number_of_flowers = len(array_geometry.index)

  # We check which bee is still foraging
  still_foraging = np.where(bee_data[:number_of_bees,1]==0.)[0]
   
  # We add another slot for the visitation sequence

  bee_route = np.concatenate((bee_route,np.full((number_of_bees,1),-1)),axis=1)

  number_of_visits = np.shape(bee_route)[1] # includes the nest at the begininng

  # Sampling next destination for each individual still foraging
  for ind in still_foraging : 
    still_foraging,bee_route = sampling_next_destination(number_of_bees,ind,bee_route,bee_data,bee_info,array_of_vector_used,number_of_flowers,learning_array_list,bout,still_foraging,array_geometry)

  # Checking if some individuals reached the same flower. If so, it triggers a competition interaction.
  flowers_visited_this_bout = bee_route[:,-1]
  flowers_in_competition = find_flowers_in_competition(flowers_visited_this_bout)

  # Competition check
  # We create a matrix saying if a bee feeds (1) or not (0). Default to 1.
  who_feeds = np.ones(number_of_bees)

  for flower in flowers_in_competition : 
    flower=int(flower)

    interaction_winner,interaction_losers = competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_info)

    for loser in interaction_losers : 

      # Set the loser's feeding potential to 0.
      who_feeds[loser] = 0

      previous_flower = bee_route[loser,-2]
      if individual_flower_outcome[loser][previous_flower,flower] == 0 : # if it is their first visit to the flower

        # Updating the negative outcome for the loser.
        individual_flower_outcome[loser][previous_flower,flower] = -1

        # If use_online_reinforcement, apply punishment now
        if use_online_reinforcement : 
          learning_array_list[loser] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[loser],previous_flower,flower,0,alpha_pos[loser],alpha_neg[loser],gamma[loser],learning_factor[loser],abandon_factor[loser])

  # If the bout is finished the bee does not feed
  who_feeds[bee_data[:number_of_bees,1]==1.] = 0
  

  # Feeding Loop
  for ind in (np.where(who_feeds==1)[0]):

    # If there was a resource available, the individual feeds. The flower is emptied
    if resources_on_flowers[bee_route[ind,-1]]==1 : 
      resources_on_flowers[bee_route[ind,-1]] = 0
      bee_data[ind,0] += 1 # number of resources foraged

    else: 
      who_feeds[ind] = 0
  
  # Increases the counter if no resource is found on the flower
  for ind in np.where(who_feeds==0)[0]:

    if bee_info["leave_after_max_fail"][ind]: 
      previous_flower = bee_route[ind,-2]
      chosen_flower = bee_route[ind,-1]
      array_of_vector_used[previous_flower,chosen_flower,ind] +=1

  # Check on passive punitive reaction (if flower was empty on first visit)
  for ind in still_foraging : 

    flower_visited = bee_route[ind,-1]

    # If this is their first visit on this flower
    if(flower_visited!=0): # Not the nest

      previous_flower = bee_route[ind,-2]
      if individual_flower_outcome[ind][previous_flower,flower_visited]==0: # Never been visited by taking this path at least. Why not using a general array with the flowers resources to keep track and see if the won the interaction on an empty flower???  

        if who_feeds[ind]==1: # Positive outcome for this flower

          if not bee_info["use_route_compare"][ind] : 
            individual_flower_outcome[ind][previous_flower,flower_visited]=1

            # If needed, apply online reinforcement here
            if use_online_reinforcement : 
              learning_array_list[ind] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[ind],previous_flower,flower_visited,1,alpha_pos[ind],alpha_neg[ind],gamma[ind],learning_factor[ind],abandon_factor[ind])           

        else : # Negative outcome for this flower 
          # Why is "route compare" not checked here?? Possible answer: route compare is only for positive RL?

          individual_flower_outcome[ind][previous_flower,flower_visited]=-1
          # If needed, apply online reinforcement here
          if use_online_reinforcement : 
            learning_array_list[ind] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[ind],previous_flower,flower_visited,0,alpha_pos[ind],alpha_neg[ind],gamma[ind],learning_factor[ind],abandon_factor[ind])   

  # Check end of foraging bout
  number_of_max_foraged_resourced_reached = (bee_data[:,0]==bee_info["max_crop"])
  bee_data[number_of_max_foraged_resourced_reached,1] = 1.
  
  # Fail safe limit of distance travelled
  for ind in range (number_of_bees) : 
    if bee_data[ind,2]>= bee_info["max_distance_travelled"][ind] : # compare distance travelled with maximum distance
      bee_data[ind,1]=1. # bout finished

  return(bee_route)


def non_online_learning_and_route_quality_update(number_of_bees,bee_route,array_geometry,bee_data,bee_info,optimal_route_quality_1_ind, optimal_route_quality_2_ind,silent_sim,array_folder,individual_flower_outcome,learning_array_list) : 
  """
  Description:
    Applies non online learning if needed and compute route qualities/
  Inputs:
    number_of_bees: number of bees
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    optimal_route_quality: estimation of optimal route quality so far
    silent_sim: if True, prevents from printing
    array_folder: name of array folder
    individual_flower_outcome: list of arrays containing the outcome of each transition for each individual
  Outputs:
    learning_array_list is modified in place
    bee_data is already modified in place
    route_qualities: list of route quality for each bee
  """ 
  use_online_reinforcement = bee_info["use_online_reinforcement"][0]
  # Initialize the output vector
  route_qualities = np.zeros((number_of_bees))
  
  # Post-bout Learning phase loop for each bee
  for ind in range (number_of_bees) : 
    if bee_info["use_delay_start"][ind] and bout<bee_info["starting_bout_for_naive"][ind] :
      route_qualities[ind]=0
      #break????

    # Get the route quality, then round it to 8 decimals (to match with the sinked value in .txt file)
    else : 
      route = geometry_functions.formatting_route(bee_route[ind,:])
      route_quality = geometry_functions.get_quality_of_route(route,array_geometry,bee_data[ind,0])
      route_quality=round(route_quality,8)

      # Check if the new route quality is higher than the optimal route found initially. If so, replace old optimal route.
      # Optimal route =/= Best route known. Optimal route is the TSP solution of the array.
      # This ensures that if the initial optimal route assessment failed (as it is assessed via simulations), any new more optimized route replaces the old one.

      if optimal_route_quality_1_ind is not None: 
        if route_quality > optimal_route_quality_1_ind : 
          optimal_route_quality_1_ind = route_quality
        
        
      # Apply the non-online learning (positive & negative) to the probability matrix.

      if not use_online_reinforcement : 
        learning_array_list[ind] = learning_functions.apply_learning(route,learning_array_list,individual_flower_outcome,bee_data,bee_info,route_quality,ind)  

      # Save the route quality of the individual
      route_qualities[ind]=route_quality

      
      if bee_info["use_route_compare"][ind] and bee_data[ind,3]<route_quality and bee_data[ind,0]==bee_data["max_crop"][ind] : 
        bee_data[ind,3]=route_quality # best route quality

  # Check if the optimal route quality for two individuals is still optimal
  if optimal_route_quality_2_ind is not None: 
    current_group_route_quality = np.sum(route_qualities) 
    current_group_route_quality=round(current_group_route_quality,8)
    if current_group_route_quality> optimal_route_quality_2_ind : 
      optimal_route_quality_2_ind = current_group_route_quality


  return(route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind)


def simulate_bout(bout,array_geometry,learning_array_list,bee_data,bee_info,optimal_route_quality_1_ind,optimal_route_quality_2_ind,silent_sim,array_folder,number_of_bees=None) : 
  """
  Description: 
    Simulates a bout 
  Inputs:
    bout:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    learning_array_list: array on which the learning process will happen. if !use_Q_learning, learning_array_list=probability_array, else learning_array_list=Q_table_list
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    bee_info: pandas dataframe of the parameters 
    optimal_route_quality: 
    silent_sim: bool, if True, prevents from printing
    use_Q_learning: bool, if True will use Q Learning instead of the learning model by T.Dubois
  Outputs: 
    learning_array_list and bee_data are modified in place
    competitive_bout: dictionary with keys: 
                            "sequences" giving the routes of the bees
                            "quality" giving the qualities of the routes
                            "optimal_route_quality" giving the optimal route quality computed so far
  """
  # Retrieve parameters of the simulation
  if number_of_bees is None : 
    number_of_bees,_ = np.shape(bee_data)
  number_of_flowers = len(array_geometry.index)

  # Initiialize data
  bee_route = np.zeros((number_of_bees,1)).astype(int)
  resources_on_flowers = np.ones(number_of_flowers)
  resources_on_flowers[0] = 0 # no resource in nest
  individual_flower_outcome = [np.zeros((number_of_flowers,number_of_flowers)) for bee in range (number_of_bees)]

  array_of_vector_used = np.zeros((number_of_flowers,number_of_flowers,number_of_bees)) # only used if leave_after_max_fail

  # use_delay_start check : if bee's starting_bout_for_naive is not reached for said ind, finish its bout instantly.

  for ind in range (number_of_bees) : 
    if bee_info["use_delay_start"][ind] and bout < bee_info["starting_bout_for_naive"][ind] :
      bee_data[ind,1] = 1.

  # Foraging loop: while at least one individual hasn't finished foraging
  while(np.sum(bee_data[:,1])!=number_of_bees): 
    bee_route=foraging_and_online_learning_loop(number_of_bees,bee_route,bee_data,bee_info,array_of_vector_used,learning_array_list,individual_flower_outcome,array_geometry,resources_on_flowers,bout)

  # Learning loop: 
  route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind = non_online_learning_and_route_quality_update(number_of_bees, bee_route, array_geometry, bee_data,bee_info, optimal_route_quality_1_ind, optimal_route_quality_2_ind, silent_sim,array_folder, individual_flower_outcome, learning_array_list)

  competitive_bout = {"sequences":bee_route, "route_quality": route_qualities, "optimal_route_quality_1_ind":optimal_route_quality_1_ind,"optimal_route_quality_2_ind":optimal_route_quality_2_ind}


  return(competitive_bout)