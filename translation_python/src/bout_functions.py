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

def give_omitted_destinations(ind,current_pos,previous_pos,array_of_vector_used,bee_data) :
  """
  Description:
    Returns the destinations that must be omitted when sampling the next destination
  Inputs:
    ind: index of the individual
    current_pos: curent position of the individual
    previous_pos: previous position of the individual
    array_of_vector_used: array giving the number of times a transition was used by a bee since the beginning of the bout (size number_of_flowers*number_of_flowers*number_of_bee)
    bee_data: pandas dataframe collecting information about the bees that will be updated throughout the simulation
  Outputs:
    omitted_destinations: vector of destinations to be omitted
  """ 
  omitted_destinations = [current_pos] # At least the self-loop is forbidden.

  if not (bee_data["allow_nest_return"][ind]) : # If condition, omit the nest.
    omitted_destinations.append(0)

  if bee_data["forbid_reverse_vector"][ind] and previous_pos is not None : # If condition, omit the previous visit.
    omitted_destinations.append(previous_pos)

  if bee_data["leave_after_max_fail"][ind] : 
    omitted_destinations=np.concatenate((omitted_destinations,np.where(array_of_vector_used[current_pos,:,ind]==bee_data["max_fails"][ind])[0]))

  omitted_destinations = np.unique(omitted_destinations) # Make sure each flower is mentioned only once.

  return(omitted_destinations)


def get_probabilities_of_choosing_flowers(use_Q_learning,learning_array_list,ind,current_pos,destination_is_available,dynamic_beta,beta_QL_vector,bout,beta_QL) :
  """
  Description:
    Give the probability of choosing among availabl destinations
  Inputs:
    use_Q_learning: if True will use a softmax function to give probabilities
    learning_array_list: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
    ind: index of individual
    current_pos: current position of individual
    destination_is_available: bool list such as destination_is_available[flower_ID] says if flower with ID flower_ID is amng the possible destinations
    dynamic_beta: bool, if True will use a dynamic beta parameter whose values are specified in beta_QL_vector
    beta_QL_vector: specifies the values of beta in case of dynamic beta
    bout: index of bout
    beta_QL: specifies inverse temperature parameter if use_Q_Learning
  Outputs:
    probabilities: a vector of probabilities
  """  
  if not use_Q_learning : 
    probabilities = learning_array_list[ind][current_pos,destination_is_available]
    
    sum_of_probabilities = np.sum(learning_array_list[ind][current_pos,destination_is_available])
    if sum_of_probabilities !=0 : 
      probabilities = probabilities/sum_of_probabilities

  else : 
    if dynamic_beta : 
      probabilities = learning_functions.softmax(learning_array_list[ind][current_pos,destination_is_available],beta_QL_vector[bout])

    else : 
      probabilities = learning_functions.softmax(learning_array_list[ind][current_pos,destination_is_available],beta_QL)
  return(probabilities)


def sampling_next_destination(number_of_bees,ind,bee_route,bee_data,array_of_vector_used,number_of_flowers,learning_array_list,bout,still_foraging,array_geometry): 
  """
  Description:
    Samples the next destination for an individual and updates the corresponding data
  Inputs:
    ind: index of individual
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    array_of_vector_used: array giving the number of times a transition was used by a bee since the beginning of the bout (size number_of_flowers*number_of_flowers*number_of_bee)
    number_of_flowers: ottal number of flowers + nest
    learning_array_list: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
    bout: index of bout
    still_foraging: vactor of indices of individuals that are still foraging
  Outputs:
    Updated still_foraging, bee_route. bee_data is already modified in place
  """ 
  # Retrieve parameters of the simulation
  beta_QL = bee_data["beta_QL"][0]
  beta_QL_vector = bee_data["beta_QL_vector"][0]
  use_Q_learning = bee_data["use_Q_learning"][0]
  leave_after_max_fail = bee_data["leave_after_max_fail"][0]
  dynamic_beta = bee_data["dynamic_beta"][0]
  online_reinforcement = bee_data["online_reinforcement"][0]

  number_of_flowers = len(array_geometry.index)
  number_of_visits = np.shape(bee_route)[1]

  # Retrieve the bee's current position
  current_pos = bee_route[ind,number_of_visits-2]

  if number_of_visits>2 : 
    previous_pos = bee_route[ind,number_of_visits-3]
  else : 
    previous_pos = None

  # Mark all destinations which must be omitted
  omitted_destinations = give_omitted_destinations(ind,current_pos,previous_pos,array_of_vector_used,bee_data)
  # Retrieve all potential destinations
  potential_destinations = np.array([flower for flower in range (number_of_flowers)])
  for omitted_destination in omitted_destinations : 
    potential_destinations = np.delete(potential_destinations, np.where(potential_destinations==omitted_destination))
  destination_is_available = [not (flower in omitted_destinations) for flower in range (number_of_flowers)]
  # Define probabilities
  probabilities = get_probabilities_of_choosing_flowers(use_Q_learning,learning_array_list,ind,current_pos,destination_is_available,dynamic_beta,beta_QL_vector,bout,beta_QL)
  # If no positive probabilities
  if (np.sum(probabilities)==0) : 
    if current_pos == 0 : # The bee was already in the nest: triggering end of bout
      bee_data.loc[ind, 'bout_finished'] = True
      still_foraging = np.delete(still_foraging,np.where(still_foraging==ind))
      next_pos = 0
    else : 
      next_pos = 0
      bee_route[ind,-1] = 0 # Go back to nest (not trigerring end of bout). Useless because it was already at 0

  else : # There is a potential destination
    if len(potential_destinations)>1 :
      next_pos = np.random.choice(a=potential_destinations,p=probabilities)
      bee_route[ind,-1] = next_pos
    else : 
      next_pos = potential_destinations[0] # Not sure why it is useful though
      bee_route[ind,-1] = next_pos

  # Update distance travelled
  bee_data.loc[ind, 'distance_travelled'] += geometry_functions.distance(array_geometry.loc[current_pos,'x':'y'],array_geometry.loc[next_pos,'x':'y'])
  # Check if the bee chose the nest
  if(bee_data["allow_nest_return"][ind]) : 
    if (bee_route[ind,-1]==0) and (bee_data["distance_travelled"][ind]>0) : 
      bee_data.loc[ind, 'bout_finished'] = True
  return(still_foraging,bee_route)


def get_probability_of_winning(individuals_in_competition,bee_data):
  """
  Description:
    Normalizes the probabilities of winning of competing individuals
  Inputs:
    individuals_in_competition: indices of competing individuals
    bee_data: pandas dataframe with data about the bees that will be updated during the simulation
  Outputs:
    A vector of probabilities 
  """ 
  not_normalized_prob = []
  for ind in individuals_in_competition : 
    not_normalized_prob.append(bee_data["probability_of_winning"][ind])
  return(np.array(not_normalized_prob)/np.sum(np.array(not_normalized_prob)))


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


def competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_data):
  """
  Description:
    Gives the winner and losers of a competing interaction
  Inputs:
    flower: index of flower
    flowers_visited_this_bout: list of flowers visited this bout
    bee_data: pandas datafram with information about bees that will be updated throughout the simulation
  Outputs:
    interaction_winner: ID of winner
    interaction_losers: list of ID of losers
  """ 
  # Which individuals are in competition
  individuals_in_competition = np.where (np.array(flowers_visited_this_bout) == flower)[0]
  probability_of_winning  = get_probability_of_winning(individuals_in_competition,bee_data)
  # Which wins and which loses
  interaction_winner = np.random.choice(a=individuals_in_competition,p=probability_of_winning)
  interaction_losers = np.delete(individuals_in_competition,np.where(individuals_in_competition==interaction_winner))
  return(interaction_winner,interaction_losers)


def foraging_and_online_learning_loop(number_of_bees,bee_route,bee_data,array_of_vector_used,learning_array_list,individual_flower_outcome,array_geometry,list_of_bout_resources,resources_on_flowers,bout):
  """
  Description:
    Each bee will decide on wht flower it will go next and will be either rewarded or punished.
  Inputs:
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    array_of_vector_used: if bee_data["leave_after_max_fail"], will storing for each vector flower_a,flower_b and for each individual the number of times the transition between flower_a and flower_b led to a negative outcome
    learning_array_list: list of arrays on which the learning process will occur (either probability matrix or Q table)
    individual_flower_outcome: list of arrays containing the outcome of each transition for each individual
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    list_of_bout_resources: list of list of individual who fed during each bout (0 if not, 1 if they fed)
    resources_on_flowers: list of resources on each flower
  Outputs:
    updated bee_route
    list_of_bout_resources, resources_on_flowers, learning_array_list, array_of_vector_used, individual_flower_outcome, bee_data are also modified in place
  """ 

  # Retrieve parameters of the simulation
  alpha_pos = bee_data["alpha_pos"][0]
  alpha_neg = bee_data["alpha_neg"][0]
  gamma_QL = bee_data["gamma_QL"][0]
  use_Q_learning = bee_data["use_Q_learning"][0]
  online_reinforcement = bee_data["online_reinforcement"][0]
  #print("online_reinforcement: ",online_reinforcement)
  cost_of_flying = bee_data["cost_of_flying"][0]
  learning_factor = bee_data["learning_factor"][0]
  abandon_factor = bee_data["abandon_factor"][0]
  leave_after_max_fail = bee_data["leave_after_max_fail"][0]

  number_of_flowers = len(array_geometry.index)

  # We check which bee is still foraging
  still_foraging = np.where(bee_data.loc[:(number_of_bees-1),"bout_finished"]==False)[0]
  #print('still foraging: ',still_foraging)
   
  # We add another slot for the visitation sequence
  bee_route = np.concatenate((bee_route,np.zeros((number_of_bees,1))),axis=1).astype(int)
  number_of_visits = np.shape(bee_route)[1] # includes the nest at the begininng
  # Sampling next destination for each individual still foraging
  for ind in still_foraging : 
    still_foraging,bee_route = sampling_next_destination(number_of_bees,ind,bee_route,bee_data,array_of_vector_used,number_of_flowers,learning_array_list,bout,still_foraging,array_geometry)
    #print('bee route: ', bee_route[ind])

  # Checking if some individuals reached the same flower. If so, it triggers a competition interaction.
  flowers_visited_this_bout = bee_route[:,-1]
  flowers_in_competition = find_flowers_in_competition(flowers_visited_this_bout)
  #print("flowers_in_competition: ", flowers_in_competition)

  # Competition check
  # We create a matrix saying if a bee feeds (1) or not (0). Default to 1.

  who_feeds = np.ones(number_of_bees)
  for flower in flowers_in_competition : 
    flower=int(flower)
    # Determine the winner and the losers of this interaction
    interaction_winner,interaction_losers = competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_data)
    #print('winner and losers: ',interaction_winner,interaction_losers)

    for loser in interaction_losers : 
      #print('punishing loser: ',loser)
      # Set the loser's feeding potential to 0.
      who_feeds[loser] = 0

      # Updating the negative outcome for the loser.
      previous_flower = bee_route[loser,-2]
      individual_flower_outcome[loser][previous_flower,flower] = -1

      # If online_reinforcement, apply punishment now
      if online_reinforcement : 
        #print('online punishment')
        #print('value of vector before learning: ',learning_array_list[loser][previous_flower,flower])
        learning_array_list[loser] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[loser],previous_flower,flower,0,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)
        #print('value of vector after learning: ',learning_array_list[loser][previous_flower,flower])
  # If the bout is finished the bee does not feed
  who_feeds[bee_data.loc[:(number_of_bees-1),"bout_finished"]] = 0
  
  # Feeding Loop
  for ind in (np.where(who_feeds==1)[0]):

    # If there was a resource available, the individual feeds. The flower is emptied
    if resources_on_flowers[bee_route[ind,-1]]==1 : 
      resources_on_flowers[bee_route[ind,-1]] = 0
      bee_data.loc[ind,"number_of_resources_foraged"]   += 1

    else: 
      who_feeds[ind] = 0

  # Update whoFeeds output
  list_of_bout_resources.append(who_feeds)
  
  # Increases the counter if no resource is found on the flower
  #print('updating leave_after_max_fail')
  for ind in np.where(who_feeds==0)[0]:
    #print("checking unsuccessful_individual: ",ind)
    if leave_after_max_fail: 
      array_of_vector_used[bee_route[ind,-2],bee_route[ind,-1],ind] +=1

  # Check on passive punitive reaction (if flower was empty on first visit)
  #print('passive punitive reaction')
  for ind in still_foraging : 
    #print('individual '+str(ind)+'is still foraging')
    flower_visited = bee_route[ind,-1]

    # If this is their first visit on this flower
    if(flower_visited!=0): # Not the nest

      #print('flower visited is not the nest')
      previous_flower = bee_route[ind,-2]
      #print('vector just used by the bee: ',previous_flower, ' -> ',flower_visited)
      #print('outcome for this vector: ',individual_flower_outcome[ind][previous_flower,flower_visited])

      if individual_flower_outcome[ind][previous_flower,flower_visited]==0: # Never been visited by taking this path at least. Why not using a general array with the flowers resources to keep track and see if the won the interaction on an empty flower???  

        if who_feeds[ind]==1: # Positive outcome for this flower
          #print("positive outcome")
          if not bee_data["route_compare"][ind] : 
            individual_flower_outcome[ind][previous_flower,flower_visited]=1

            # If needed, apply online reinforcement here
            if online_reinforcement : 
              #print('value of vector before learning: ',learning_array_list[ind][previous_flower,flower_visited])
              learning_array_list[ind] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[ind],previous_flower,flower_visited,1,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)           
              #print('value of vector after learning: ',learning_array_list[ind][previous_flower,flower_visited])
        else : # Negative outcome for this flower 
          #print("negative outcome")
        # Why is "route compare" not checked here??
          individual_flower_outcome[ind][previous_flower,flower_visited]=-1
          # If needed, apply online reinforcement here
          if online_reinforcement : 
            #print('value of vector before learning: ',learning_array_list[ind][previous_flower,flower_visited])
            learning_array_list[ind] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array_list[ind],previous_flower,flower_visited,0,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)   
            #print('value of vector after learning: ',learning_array_list[ind][previous_flower,flower_visited])

  # Check end of foraging bout
  number_of_max_foraged_resourced_reached = bee_data["number_of_resources_foraged"]==bee_data["max_crop"]
  bee_data.loc[number_of_max_foraged_resourced_reached,"bout_finished"]=True
  #if np.sum(number_of_max_foraged_resourced_reached)!=0:
    #print("############ bout finished for individuals ", np.where(number_of_max_foraged_resourced_reached)[0]," because their crops are full.")
  
  # Fail safe limit of distance travelled
  for ind in range (number_of_bees) : 
    if bee_data["distance_travelled"][ind]>= bee_data["max_distance_travelled"][ind] : 
      #print("########### bout finished for individual "+str(ind)+" because it has reached its maximum travelled distance.")
      bee_data.loc[ind,"bout_finished"]=True

  #print('individual_flower_outcome: ',individual_flower_outcome)
  #print('bee_route: ',bee_route)
  #print('list_of_bout_resources: ',list_of_bout_resources)
  #print('resources_on_flowers: ', resources_on_flowers)
  #print('learning_array_list: ', learning_array_list)
  #print('array_of_vector_used: ',array_of_vector_used)


  return(bee_route)


def non_online_learning_and_route_quality_update(number_of_bees,bee_route,array_geometry,bee_data,optimal_route_quality, silent_sim,array_folder,individual_flower_outcome,learning_array_list) : 
  """
  Description:
    Applies non online learning if needed and compute route qualities/
  Inputs:
    number_of_bees: number of bees
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    optimal_route_quality: estimation of optimal route quality so far
    silent_sim: if True, prevents from printing
    array_folder: name of array folder
    individual_flower_outcome: list of arrays containing the outcome of each transition for each individual
  Outputs:
    learning_rray_list is modified in place
    bee_data is already modified in place
    route_qualities: list of route quality for each bee
  """ 
  online_reinforcement = bee_data["online_reinforcement"][0]
  # Initialize the output vector
  route_qualities = np.zeros((number_of_bees))
  
  # Post-bout Learning phase loop for each bee
  for ind in range (number_of_bees) : 
    if bee_data["different_experience_simulation"][ind] and bout<bee_data["starting_bout_for_naive"][ind] :
      #print("bout not started for ind ",ind) 
      route_qualities[ind]=0
      #print("route qualities",route_qualities)
      #break #why put a break here???

    # Get the route quality, then round it to 8 decimals (to match with the sinked value in .txt file)
    else : 
      #print("bout had started for individual ",ind) 
      route = geometry_functions.formatting_route(bee_route[ind,:])
      route_quality = geometry_functions.get_route_quality(route,array_geometry,bee_data["number_of_resources_foraged"][ind])
      route_quality=round(route_quality,8)

      # Check if the new route quality is higher than the optimal route found initially. If so, replace old optimal route.
      # Optimal route =/= Best route known. Optimal route is the TSP solution of the array.
      # This ensures that if the initial optimal route assessment failed (as it is assessed via simulations), any new more optimized route replaces the old one.

      if optimal_route_quality is not None: 
        if route_quality > optimal_route_quality : 
          if not silent_sim: 
            print("The following route ended with superior quality than optimal : ", route)
          optimal_route_quality = route_quality
          pd.DataFrame({"optimal_route":[optimal_route_quality]}).to_csv(path_or_buf = array_folder+'\\optimal_route.csv', index = False)
        
    
      # Apply the non-online learning (positive & negative) to the probability matrix.

      if not online_reinforcement : 
        #print("reinforcement not online. probability_matrix of individual "+str(ind)+" before learning: ")
        #print(learning_array_list[ind])
        learning_array_list[ind] = learning_functions.apply_learning(route,learning_array_list[ind],individual_flower_outcome[ind],bee_data,route_quality)
        #print("reinforcement not online. probability_matrix of individual "+str(ind)+" after learning: ")
        #print(learning_array_list[ind])     

      # Save the route quality of the individual
      route_qualities[ind]=route_quality
      #print("route qualities",route_qualities)
      
      if bee_data["route_compare"][ind] and bee_data["best_route_quality"][ind]<route_quality and bee_data["number_of_resources_foraged"][ind]==bee_data["max_crop"][ind] : 
        bee_data["best_route_quality"][ind]=route_quality

  return(route_qualities)


def competitive_route(bout,array_geometry,learning_array_list,bee_data,optimal_route_quality,silent_sim,array_folder,number_of_bees=None) : 
  """
  Description: 
    Simulates a bout 
  Inputs:
    bout:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    learning_array_list: array on which the learning process will happen. if !use_Q_learning, learning_array_list=probability_array, else learning_array_list=Q_table_list
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation
    optimal_route_quality: 
    silent_sim: bool, if True, prevents from printing
    use_Q_learning: bool, if True will use Q Learning instead of the learning model by T.Dubois
  Outputs: 
    learning_array_list and bee_data are modified in place
    competitive_bout: dictionary with keys: 
                            "sequences" giving the routes of the bees
                            "quality" giving the qualities of the routes
                            "optimal_route_quality" giving the optimal route quality computed so far
                            "list_of_bout_resources" 
  """

  # Retrieve parameters of the simulation
  if number_of_bees is None : 
    number_of_bees = len(bee_data.index)
  number_of_flowers = len(array_geometry.index)

  leave_after_max_fail = bee_data.loc[0,"leave_after_max_fail"]

  # Initiialize data
  bee_route = np.zeros((number_of_bees,1)).astype(int)
  resources_on_flowers = np.ones(number_of_flowers)
  resources_on_flowers[0] = 0 # no resource in nest
  individual_flower_outcome = [np.zeros((number_of_flowers,number_of_flowers)) for bee in range (number_of_bees)]
  list_of_bout_resources = []

  array_of_vector_used = np.zeros((number_of_flowers,number_of_flowers,number_of_bees))

  # different_experience_simulation check : if bee's starting_bout_for_naive is not reached for said ind, finish its bout instantly.
  for ind in range (number_of_bees) : 
    if bee_data["different_experience_simulation"][ind] and bout < bee_data["starting_bout_for_naive"][ind] :
      #print("############## individual "+str(ind)+" is not foraging because its starting bout has not occured yet")
      bee_data["bout_finished"] = True

  # Foraging loop: while at least one individual hasn't finished foraging
  while(np.sum(bee_data["bout_finished"])!=number_of_bees): 
    #print('number of bees that finished: ', np.sum(bee_data["bout_finished"]))
    bee_route=foraging_and_online_learning_loop(number_of_bees,bee_route,bee_data,array_of_vector_used,learning_array_list,individual_flower_outcome,array_geometry,list_of_bout_resources,resources_on_flowers,bout)

  # Learning loop: 
  route_qualities = non_online_learning_and_route_quality_update(number_of_bees, bee_route, array_geometry, bee_data, optimal_route_quality, silent_sim,array_folder, individual_flower_outcome, learning_array_list)


  bee_route = np.concatenate((bee_route,np.zeros((number_of_bees,1))),axis=1).astype(int) # For formatting
  competitive_bout = {"sequences":bee_route, "route_quality": route_qualities, "optimal_route_quality":optimal_route_quality,"list_of_bout_resources":list_of_bout_resources}
  return(competitive_bout)




"""
alpha_pos = 0.5
alpha_neg = 0.2
gamma_QL = 0
cost_of_flying = False
learning_factor = 1.5
abandon_factor = 0.75
leave_after_max_fail = True

different_experience_simulation = False
starting_bout_for_naive=[0,0]

online_reinforcement = True
use_Q_learning = False
dynamic_beta=True
bout=0
beta_QL=2
beta_QL_vector=[[[2,2]]]
array_number = 1

current_pos = 1
previous_pos = 0
number_of_bees = 1
number_of_flowers = 3
array_of_vector_used = np.zeros((number_of_flowers,number_of_flowers,number_of_bees))

optimal_route_quality = 0
silent_sim = False

import spatial_array_generation_and_manipulation_functions
array_info = {'environment_type':'generate', 'number_of_flowers' : number_of_flowers-1, 'number_of_patches' : 1, 'patchiness_index' : 0, 'env_size' : 500, 'flowers_per_patch' : None }
array_geometry, array_info_new, array_folder = spatial_array_generation_and_manipulation_functions.create_environment (array_info, array_number,True,False)
print('array geometry ',array_geometry)


import management_of_data_functions as man
param_tracking = {"number_of_resources_foraged" : 0,"bout_finished": False,"distance_travelled":0.}
param_indiv = {"starting_bout_for_naive":starting_bout_for_naive,"different_experience_simulation":different_experience_simulation,"max_fails":2,"route_compare":False,"max_distance_travelled": 10000,"max_crop":5,"probability_of_winning":0.5,"dynamic_beta":dynamic_beta,"beta_QL_vector":beta_QL_vector,"leave_after_max_fail": leave_after_max_fail, "max_fails":2,"forbid_reverse_vector": True,"allow_nest_return": False,"beta_QL": beta_QL,"alpha_pos":alpha_pos,"alpha_neg":alpha_neg,"gamma_QL":gamma_QL,"use_Q_learning":use_Q_learning,"online_reinforcement": online_reinforcement,"cost_of_flying":cost_of_flying,"learning_factor":learning_factor,"abandon_factor":abandon_factor}
bee_data = man.initialize_bee_data(number_of_bees,param_tracking,param_indiv)

learning_array_list = learning_functions.initialize_Q_table_list ("distance", array_geometry, 2, number_of_bees)

print(learning_array_list)
competitive_bout = competitive_route(bout,array_geometry,learning_array_list,bee_data,optimal_route_quality,silent_sim,array_folder,number_of_bees=None)
print(learning_array_list)

print(competitive_bout["sequences"])
"""