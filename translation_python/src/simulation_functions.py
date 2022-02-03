'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import os
import pandas as pd
import parameters
import learning_functions
import geometry_functions
import spatial_array_generation_and_manipulation_functions

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


def get_probabilities_of_choosing_flower(use_Q_learning,learning_array,ind,current_pos,destination_is_available,dynamic_beta,beta_QL_vector,bout,beta_QL) :
  """
  Description:
    Give the probability of choosing among availabl destinations
  Inputs:
    use_Q_learning: if True will use a softmax function to give probabilities
    learning_array: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
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
    probabilities = learning_array[ind][current_pos,destination_is_available]

  else : 
    if dynamic_beta : 
      probabilities = learning_functions.softmax(learning_array[ind][current_pos,destination_is_available],beta_QL_vector[bout])

    else : 
      probabilities = learning_functions.softmax(learning_array[ind][current_pos,destination_is_available],beta_QL)
  return(probabilities)


def sampling_next_destination(ind,bee_route,bee_data,array_of_vector_used,number_of_flowers,learning_array,bout,still_foraging): 
  """
  Description:
    Samples the next destination for an individual and updates the corresponding data
  Inputs:
    ind: index of individual
    bee_route: matrix of size number_of_bees*number_of_visits_so_far(including the visit that is currently sampled), storing the routes of ech bee so far
    bee_data: pandas dataframe of relvant data about the bees that will be updated during the simulation
    array_of_vector_used: array giving the number of times a transition was used by a bee since the beginning of the bout (size number_of_flowers*number_of_flowers*number_of_bee)
    number_of_flowers: ottal number of flowers + nest
    learning_array: list of learning arrays (either probability matrix if not use_Q_learning or Q_table otherwise) for each individual
    bout: index of bout
    still_foraging: vactor of indices of individuals that are still foraging
  Outputs:
    Updated still_foraging, bee_route and bee_data
  """ 

  # Retrieve parameters of the simulation
  beta_QL = bee_data["beta_QL"][0]
  beta_QL_vector = bee_data["beta_QL_vector"][0]
  use_Q_learning = bee_data["use_Q_learning"][0]
  leave_after_max_fail = bee_data["leave_after_max_fail"][0]
  dynamic_beta = bee_data["dynamic_beta"][0]
  online_reinforcement = bee_data["online_reinforcement"][0]

  number_of_bees = len(bee_data.index)
  number_of_flowers = len(array_geometry.index)
  number_of_visits = np.shape(bee_route)[1]

  # Retrieve the bee's current position
  current_pos = int(bee_route[ind,number_of_visits-2])

  if number_of_visits>2 : 
    previous_pos = int(bee_route[ind,number_of_visits-3])
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
  probabilities = get_probabilities_of_choosing_flower(use_Q_learning,learning_array,ind,current_pos,destination_is_available,dynamic_beta,beta_QL_vector,bout,beta_QL)

  # If no positive probabilities
  if (np.sum(probabilities)==0) : 

    if current_pos == 0 : # The bee was already in the nest: triggering end of bout
      bee_data["bout_finished"][ind] = True
      still_foraging = np.delete(still_foraging,np.where(still_foraging==ind))

    else : 
      bee_route[ind,-1] = 0 # Go back to nest (not trigerring end of bout). Useless because it was already at 0

  else : # There is a potential destination

    if len(potential_destinations)>1 :
      next_pos = np.random.choice(a=potential_destinations,p=probabilities)
    else : 
      next_pos = potential_destinations[0] # Not sure why it is useful though
    
    bee_route[ind,-1] = next_pos

  # Update distance travelled
  bee_data["distance_travelled"][ind] += geometry_functions.distance(array_geometry.iloc[current_pos,1:3],array_geometry.iloc[next_pos,1:3])
  #print("distance between flowers: ", geometry_functions.get_matrix_of_distances_between_flowers(array_geometry))
  # Check if the bee chose the nest
  if(bee_data["allow_nest_return"][ind]) : 
    if (bee_route[ind,-1]==0) and (bee_data["distance_travelled"]>0) : 
      bee_data["bout_finished"][ind] = True

  return(still_foraging,bee_route,bee_data)


def get_probability_of_winning(individuals_in_competition,bee_data):
  """
  Description:
  Inputs:
  Outputs:
  """ 
  not_normalized_prob = []
  for ind in individuals_in_competition : 
    not_normalized_prob.append(bee_data["win_probability"][ind])
  return(np.array(not_normalized_prob)/np.sum(np.array(not_normalized_prob)))


def find_flowers_in_competition(flowers_visited_this_bout):
  """
  Description:
  Inputs:
  Outputs:
  """ 
  unique_flowers, count_occurences = np.unique(flowers_visited_this_bout, return_counts=True)
  flowers_in_competition = unique_flowers[count_occurences>1]
  
  # We exclude potential detections of nest.
  flowers_in_competition = np.delete(flowers_in_competition,np.where(flowers_in_competition==0))
  return(flowers_in_competition)


def competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_data):
  """
  Description:
  Inputs:
  Outputs:
  """ 
  # Which individuals are in competition
  individuals_in_competition = np.where (flowers_visited_this_bout == flower)[0]
  probability_of_winning  = get_probability_of_winning(individuals_in_competition,bee_data)
  # Which wins and which loses
  interaction_winner = np.random.choice(a=individuals_in_competition,p=probability_of_winning)
  interaction_losers = np.delete(individuals_in_competition,np.where(individuals_in_competition==interaction_winner))
  return(interaction_winner,interaction_losers)


def foraging_loop(bee_route,bee_data,array_of_vector_used,learning_array,individual_flower_outcome,array_geometry,list_of_bout_resources,resources_on_flowers):
  """
  Description:
  Inputs:
  Outputs:
  """ 

  # Retrieve parameters of the simulation
  alpha_pos = bee_data["alpha_pos"][0]
  alpha_neg = bee_data["alpha_ng"][0]
  gamma_QL = bee_data["gamma_QL"][0]
  use_Q_learning = bee_data["use_Q_learning"][0]
  online_reinforcement = bee_data["online_reinforcement"][0]
  cost_of_flying = bee_data["cost_of_flying"][0]
  learning_factor = bee_data["learning_factor"][0]
  abandon_factor = bee_data["abandon_factor"][0]
  leave_after_max_fail = bee_data["leave_after_max_fail"][0]

  number_of_bees = len(bee_data.index)
  number_of_flowers = len(array_geometry.index)

  # We check which bee is still foraging
  still_foraging = np.where(bee_data["bout_finished"]==False)[0]
   
  # We add another slot for the visitation sequence
  bee_route = np.concatenate((bee_route,np.zeros(number_of_bees,1)),axis=1)
  number_of_visits = np.shape(bee_route)[1] # includes the nest at the begininng
 
  # Sampling next destination for each individual still foraging
  for ind in still_foraging : 
    still_foraging,bee_route,bee_data=sampling_next_destination(ind,bee_route,bee_data,array_of_vector_used,number_of_flowers,learning_array,bout,still_foraging)

  # Checking if some individuals reached the same flower. If so, it triggers a competition interaction.
  flowers_visited_this_bout = bee_route[:,-1]
  flowers_in_competition = find_flowers_in_competition(flowers_visited_this_bout)
  
  # Competition check
  # We create a matrix saying if a bee feeds (1) or not (0). Default to 1.

  who_feeds = np.ones(number_of_bees)
  for flower in flowers_in_competition : 

    # Determine the winner and the losers of this interaction
    interaction_winner,interaction_losers = competitive_interaction_on_flower(flower,flowers_visited_this_bout,bee_data)

    for loser in interaction_losers : 
      # Set the loser's feeding potential to 0.
      who_feeds[loser] = 0

      # Updating the negative outcome for the loser.
      previous_flower = bee_data[loser,-2]
      individual_flower_outcome[loser][previous_flower,flower] = -1

      # If online_reinforcement, apply punishment now
      if online_reinforcement : 
        learning_array[loser] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array[loser],previous_flower,flower,0,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)

  # If the bout is finished the bee does not feed
  who_feeds[bee_data["bout_finished"]] = 0
  
  # Feeding Loop
  for ind in (np.where(who_feeds==1)[0]):

    # If there was a resource available, the individual feeds. The flower is emptied
    if resources_on_flowers[bee_route[ind,-1]]==1 : 
      resources_on_flowers[bee_route[ind,-1]] = 0
      bee_data["number_of_resources_foraged"][ind] += 1

    else: 
      who_feeds[ind] = 0

  # Update whoFeeds output
  list_of_bout_resources.append(who_feeds)
  
  # Increases the counter if no resource is found on the flower
  for ind in np.where(whoo_feeds==0)[0]:
    if leave_after_max_fail: 
      array_of_vector_used[bee_route[ind,-2],bee_route[ind,-1]] +=1

  # Check on passive punitive reaction (if flower was empty on first visit)
  for ind in still_foraging : 
    flower_visited = beeRoute[ind,-1]

    # If this is their first visit on this flower
    if(flower_visited!=0): # Not the nest
      previous_flower = flower_visited = beeRoute[ind,-2]
      if individual_flower_outcome[ind][previous_flower,flower_visited]==0: # Never been visited by taking this path at least. Why not using a general array with the flowers resources to keep track and see if the won the interaction on an empty flower??? 
        
        if who_feeds[ind]==1: # Positive outcome for this flower
          if not bee_data["route_compare"][ind] : 
            individual_flower_outcome[ind][previous_flower,flower_visited]=1

            # If needed, apply online reinforcement here
            if online_reinforcement : 
              learning_array[loser] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array[loser],previous_flower,flower,1,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)           

        else : # Negative outcome for this flower 
        # Why is "route compare" not checked here??
          individual_flower_outcome[ind][previous_flower,flower_visited]=-1
          # If needed, apply online reinforcement here
          if online_reinforcement : 
            learning_array[loser] = learning_functions.online_learning(cost_of_flying,array_geometry,use_Q_learning,learning_array[loser],previous_flower,flower,0,alpha_pos,alpha_neg,gamma_QL,learning_factor,abandon_factor)   
  
  # Check end of foraging bout
  bee_data["bout_finished"][which(bee_data["number_of_resources_foraged"]==bee_data["max_crop"][0])]=True
  
  # Fail safe limit of distance travelled
  for ind in range (number_of_bees) : 
    if bee_data["distance_travelled"][ind]>= bee_data["max_distance_travelled"][ind] : 
      bee_data["bout_finished"][ind]=True

  return(individual_flower_outcome, bee_data, bee_route, list_of_bout_resources, resources_on_flowers, learning_array, array_of_vector_used)


def learning_loop(number_of_bees,bee_route,array_geometry,bee_data,optimal_route_quality, silent_sim,array_folder, online_learning,individual_flower_outcome) : 
  """
  Description:
  Inputs:
  Outputs:
  """ 
  # Initialize the output vector
  route_qualities = np.zeros((number_of_bees))
  
  # Post-bout Learning phase loop for each bee
  for ind in range (number_of_bees) : 
    if bee_data["different_experience_simulation"][ind] and bout<bee_data["starting_bout_for_naive"][ind] : 
      route_qualities[ind]=0
      #break #why put a break here???

    # Get the route quality, then round it to 8 decimals (to match with the sinked value in .txt file)
    else : 
      route = geometry_functions.formating_route(bee_route[ind,:])
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
        pd.DataFrame({"optimal_route":optimal_route}).to_csv(path_or_buf = array_folder+'\\optimal_route.csv', index = False)
        
    
    # Apply the non-online learning (positive & negative) to the probability matrix.

    if not online_learning : 
      learning_array[ind] = learning_functions.apply_learning(route,learning_array[ind],individual_flower_outcome[ind],bee_data,route_quality)
      
    # Save the route quality of the individual
    route_qualities[ind]=route_quality
    
    if bee_data["route_compare"][ind] and bee_data["best_route_quality"][ind]<route_quality and bee_data["number_of_resources_foraged"][ind]==bee_data["max_crop"][ind] : 
      bee_data["best_route_quality"][ind]=route_quality
  return(bee_data,learning_array,route_qualities)


def competitive_route(bout,array_geometry,learning_array,bee_data,optimal_route_quality,silent_sim=parameters.silent_sim) : 
  """
  Description: 
    Simulates a bout 
  Inputs:
    bout:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    learning_array: array on which the learning process will happen. if !use_Q_learning, learning_array=probability_array, else learning_array=Q_table_list
    bee_data: pandas dataframe containing information about bees that will be changed throughout the simulation
    optimal_route_quality: 
    silent_sim: bool, if True, prevents from printing
    use_Q_learning: bool, if True will use Q Learning instead of the learning model by T.Dubois
  Outputs: 
    competitive_bout: dictionary with keys: 
                            "sequences" giving the routes of the bees
                            "learning" giving the updated learning_array
                            "bee_data" giving the updated bee_data
                            "quality" giving the qualities of the routes
                            "optimal_route_quality" giving the optimal route quality computed so far
                            "list_of_bout_resources" 
  """

  # Retrieve parameters of the simulation
  leave_after_max_fail = bee_data["leave_after_max_fail"][0]
  number_of_bees = len(bee_data.index)
  number_of_flowers = len(array_geometry.index)

  # Initiialize data
  bee_route = np.zeros((number_of_bees,1))
  resources_on_flowers = np.ones(number_of_flowers)
  resources_on_flowers[0] = 0 # no resource in nest
  flower_outcome = np.zeros((number_of_flowers,number_of_flowers))
  individual_flower_outcome = [flower_outcome for bee in range (number_of_bees)]
  list_of_bout_resources = []

  if leave_after_max_fail : 
    array_of_vector_used = np.zeros((number_of_flowers,number_of_flowers,number_of_bees))

  # different_experience_simulation check : if bee's starting_bout_for_naive is not reached for said ind, finish its bout instantly.
  for ind in range (number_of_bees) : 
    if bee_data["different_experience_simulation"][ind] : 
      if bout < bee_data["starting_bout_for_naive"][ind] :
        bee_data["bout_finished"] = True

  # Foraging loop: while at least one individual hasn't finished foraging
  while(np.sum(bee_data["bout_finished"])!=number_of_bees): 
    individual_flower_outcome, bee_data, bee_route, list_of_bout_resources, resources_on_flowers, learning_array, array_of_vector_used=foraging_loop(bee_route,bee_data,array_of_vector_used,learning_array,individual_flower_outcome,array_geometry,list_of_bout_resources,resources_on_flowers)

  # Learning loop: 
  bee_data, learning_array, route_qualities = learning_loop(number_of_bees, bee_route, array_geometry, bee_data, optimal_route_quality, silent_sim,array_folder, online_learning, individual_flower_outcome)

  competitive_bout = {"sequences":bee_route, "learning":learning_array, "bee_data": bee_data, "route_quality": route_qualities, "optimal_route_quality":optial_route_quality,"list_of_bout_resources":list_of_bout_resources}
  return(competitive_bout)



alpha_pos = 0.5
alpha_neg = 0.2
gamma_QL = 0
online_reinforcement = True
cost_of_flying = False
learning_factor = 1.5
abandon_factor = 0.75
leave_after_max_fail = False

use_Q_learning = True
dynamic_beta=True
bout=1
beta_QL=2
beta_QL_vector=[2,2]
array_number = 1
array_info = {'environment_type':'generate', 'number_of_resources' : 5, 'number_of_patches' : 1, 'patchiness_index' : 0, 'env_size' : 500, 'flowers_per_patch' : None }
array_geometry, array_info_new, array_folder = spatial_array_generation_and_manipulation_functions.create_environment (array_info, array_number)
print('array geometry ',array_geometry)
current_pos = 1
previous_pos = 0
number_of_bees = 1
ind=0
number_of_flowers = 6
array_of_vector_used = np.zeros((number_of_flowers,number_of_flowers,number_of_bees))

import management_of_data_functions as man
param_tracking = {"number_of_resources_foraged" : 0,"bout_finished": False,"distance_travelled":0.}
param_indiv = {"dynamic_beta":dynamic_beta,"beta_QL_vector":[beta_QL_vector],"leave_after_max_fail": leave_after_max_fail, "max_fails":2,"forbid_reverse_vector": True,"allow_nest_return": False,"beta_QL": beta_QL,"alpha_pos":alpha_pos,"alpha_neg":alpha_neg,"gamma_QL":gamma_QL,"use_Q_learning":use_Q_learning,"online_reinforcement": online_reinforcement,"cost_of_flying":cost_of_flying,"learning_factor":learning_factor,"abandon_factor":abandon_factor}
bee_data = man.initialize_bee_data(number_of_bees,param_tracking,param_indiv)

learning_array = learning_functions.initialize_Q_table_list ("zero", array_geometry, 2, number_of_bees)

bee_route = np.zeros((number_of_bees,2))
still_foraging = [0]

still_foraging,bee_route,bee_data = sampling_next_destination(ind,bee_route,bee_data,array_of_vector_used,number_of_flowers,learning_array,bout,still_foraging)














"""

#########################
## MAIN
#########################

# Get starting time of simulation to get computation time.
simStart = Sys.time();

# Create Output directory in the current working directory.
dir.create("Output",showWarnings = F)
dir.create("Arrays",showWarnings = F)

# If environmentType is not a "generate", there is no need for multiple arrays.
if(environmentType!="generate") numberOfArrays = 1;



numberOfParameterSets = 0;

# Successive loops for each parameter. Thus, all parameter combinations are tested.

for (alphaPos in alphaPosList)

  
  for (alphaNeg in alphaNegList)
   
    
    for (betaQL in betaList)
    
      for (gammaQL in gammaList)
      
        
        for(algorithmRC in param.useRouteCompare)
        
          for(learningValue in param.learningFactor)
          
            for(abandonValue in param.abandonFactor)
            
              # 4.1 - Initializing ------------------------------------------------------------
              
              numberOfParameterSets = numberOfParameterSets + 1;
              
              # Create test name according to parameter values
              if(algorithmRC) textRC = "RouteCompare" else textRC = "NoRouteCompare";
              timnw = as.character(Sys.time());timstmp = paste(substr(timnw,1,4),substr(timnw,6,7),substr(timnw,9,10),substr(timnw,12,13),substr(timnw,15,16),substr(timnw,18,19),sep="")
              testName = paste(environmentType,"-",testNameGeneral,"-ParamSet",numberOfParameterSets,"-",timstmp,sep="")
              if(!silentSim) cat("Starting simulation for test : ",testName,"\n",sep="");
              # Create the output folders
              outputFolder = paste(getwd(),"/Output/",testName,sep="");
              dir.create(outputFolder,showWarnings = F);
              
              
              
              # Complete list of individual parameters. Can be a single value or a vector if you want to attribute different values to each forager.
              # Cannot take lists to test different combinations of parameters.
              paramIndiv = list(maxDistance = maximumBoutDistance,
                                #maxCrop = (nrow(arrayGeometry)-1)/numberOfBees,
                                maxCrop = 5,
                                learningFactor = learningValue,
                                abandonFactor = abandonValue,
                                maxFails = numberOfMaxFails,
                                routeCompare = algorithmRC,
                                bestRouteQuality = 0,
                                leaveAfterMaxFail = leaveAfterMaxFail,
                                allowNestReturn = allowNestReturn,
                                forbidReverseVector = forbidReverseVector,
                                differentExperienceSimulation = differentExperienceSimulation,
                                startingBoutForNaive = startingBoutForNaive,
                                winProbabilities = rep(1/numberOfBees,times=numberOfBees),
                                useQLearning=useQLearning,
                                initializeQTable=initializeQTable,
                                alphaPos=alphaPos,
                                alphaNeg=alphaNeg,
                                beta=betaQL,
                                gamma=gammaQL)
              ;
              
              # 4.2 - Simulation --------------------------------------------------------------
              
              # Parameters tracked during the simulation for each bee
              paramTracking = list(numberOfResources = rep(0,numberOfBees),
                                   probabilityWin = rep(0,numberOfBees),
                                   boutFinished = rep(F,numberOfBees),
                                   distanceTravelled = rep(0,numberOfBees));
              
              # Initialize the parameter dataframe for each bee
              beeData = InitializeBeeData(numberOfBees,paramTracking,paramIndiv);
              
              
              
              
              for(arrayNumber in 1:numberOfArrays)
              
                # Initialize the informations of the arrays
                arrayInfos = list(environmentType = environmentType,
                                  numberOfResources = numberOfResources,
                                  numberOfPatches = numberOfPatches,
                                  patchinessIndex = patchinessIndex,
                                  envSize = envSize,
                                  flowerPerPatch = flowerPerPatch)
                
                # Generate array: See 3.1 in 02-Parameters for details.
                env = CreateEnvironment(arrayInfos,arrayNumber,reuseGeneratedArrays)
                
                # Redistribute the outputs of env in variables.
                arrayGeometry = env$arrayGeometry;
                arrayInfos = env$arrayInfos;
                arrayFolder = env$arrayFolder;
                
                print('initializing probability matrices')
                # Initialize the probability matrix
                initialProbabilityArray = GetDistFctProbability(arrayGeometry,distFactor,beeData);
                
                # Apply modifiers to matrices depending on options
                for(ind in 1:numberOfBees)
                
                  if(!beeData$allowNestReturn[ind]) initialProbabilityArray[[ind]][,1] = 0;
                
                
                # Distribute probability matrix to all bees
                initialProbabilityMatrices = list();
                for(bee in 1:numberOfBees) initialProbabilityMatrices[[bee]] = NormalizeMatrix(initialProbabilityArray[[bee]]);
                
                # Save a copy of the initial probability matrix
                write.csv(initialProbabilityMatrices[[1]],paste(arrayFolder,"/probabilityMatrix.csv",sep=""),row.names = F);
                
                print('initializing Q Tables')
                #Initialize Q Tables
                initialQTableList=InitializeQTableList(initializeQTable,arrayGeometry,distFactor,beeData);
                #Save a copy of the initial QTable
                write.csv(initialQTableList[[1]],paste(arrayFolder,"/QTable.csv",sep=""),row.names = F);
                
                # Get maximum route quality of the array (simulating 300 1Ind for 30 each bouts to try and find the optimal route).
                
                  optimalRouteQuality = NULL;
                  optimalRouteQuality = SimDetectionOptimalRoute(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality : ",optimalRouteQuality,"\n",sep="")
                
                
                # Get maximum route quality for 2 ind of the array (simulating 300 1Ind for 30 each bouts to try and find the optimal route).
                
                  optimalRouteQuality2Ind = NULL;
                  optimalRouteQuality2Ind = SimDetectionOptimalRoute2Ind(arrayInfos$arrayID,arrayGeometry,beeData,initialProbabilityMatrices,arrayFolder,silentSim,0);
                  cat("Optimal Route Quality for 2 Individuals : ",optimalRouteQuality2Ind,"\n",sep="")
                
                
                # Initialize distance matrix
                distanceMatrix = as.matrix(dist(arrayGeometry[,2:3],upper = T,diag=T)); 
                
                outputFolderSim = paste(outputFolder,"/Array",sprintf("%02d",arrayNumber),sep="");
                dir.create(outputFolderSim,showWarnings = F);
                # Create a DF of information to be retrieve in further analyses (and remember what parameters were used).
                beeInfos = BuildBeeInfos(numberOfBees,paramIndiv,arrayInfos$arrayID);
                write.csv(beeInfos,paste(outputFolderSim,"/beeInfos.csv",sep=""),row.names = F);
                
                savedArrayInfos = arrayInfos;
                arrayInfosSaved = arrayInfos;
                arrayInfosSaved$flowerPerPatch = paste(arrayInfosSaved$flowerPerPatch,collapse="");
                write.csv(as.data.frame(arrayInfosSaved),paste(outputFolderSim,"/arrayInfos.csv",sep=""),row.names = F);
                write.csv(arrayGeometry,paste(outputFolderSim,"/arrayGeometry.csv",sep=""),row.names = F);
                
                # Initialize the output structures
                listOfVisitationSequences = list();
                listOfResourcesTaken = list();
                arrayOfMatrixDistance = data.frame(sim = rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees),
                                                   bout = rep(1:numberOfBouts,times = numberOfSimulations, each=numberOfBees),
                                                   bee = rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts),
                                                   distance = numeric(numberOfSimulations*numberOfBouts*numberOfBees));
                matrixOfBeeData = matrix(0,ncol=5,nrow=numberOfSimulations*numberOfBouts*numberOfBees);
                matrixOfBeeData[,1] = rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees);
                matrixOfBeeData[,2] = rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees);
                matrixOfBeeData[,3] = rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts);
                
                i = 1;
                j = 1;
                # Sim loop
                for(sim in 1:numberOfSimulations)
                
                  # Initialize simulation objects
                  beeSequences = list();
                  if(!useQLearning) learningArray=initialProbabilityMatrices else learningArray=initialQTableList;
                  # Bout loop
                  for(bout in 1:numberOfBouts)
                  
                    beeData = RebootBeeData(beeData);
                    currentBout = CompetitiveRoute(bout,arrayGeometry,learningArray,beeData,optimalRouteQuality,silentSim,useQLearning);
                    # For Sensitivity Analysis, compare learningArray and currentBout$Learning
                    for(bee in 1:numberOfBees)
                    
                      previousMatrix = learningArray[[bee]];
                      nextMatrix = currentBout$Learning[[bee]];
                      
                      arrayOfMatrixDistance$distance[i+(bee-1)] = sum(abs(previousMatrix - nextMatrix));
                    
                    
                    ## Update variables
                    learningArray = currentBout$Learning;
                    beeData = currentBout$beeData;
                    optimalRouteQuality = currentBout$optimalRouteQuality;
                    
                    matrixOfBeeData[i:(i+numberOfBees-1),4] = beeData$numberOfResources;
                    matrixOfBeeData[i:(i+numberOfBees-1),5] = currentBout$quality;
                    
                    listOfVisitationSequences[[(sim-1)*numberOfBouts+bout]] = currentBout$Sequences;
                    
                    i = i + numberOfBees;
                    
                    listOfResourcesTaken[[j]] = currentBout$listOfBoutResources;
                    j = j + 1;
                  
                
                
                # 4.3 - Formatting Raw Data ------------------------------------------
                maxLength = 0;
                for(visitSeq in 1:length(listOfVisitationSequences))
                
                  maxLength = max(maxLength,dim(listOfVisitationSequences[[visitSeq]])[2]);
                
                i=0;
                matrixOfVisitationSequences = matrix(0,nrow=numberOfBouts*numberOfSimulations*numberOfBees,ncol=maxLength+2);
                for(bout in 1:length(listOfVisitationSequences))
                
                  for(bee in 1:numberOfBees)
                  
                    i = i + 1;
                    seqBee = listOfVisitationSequences[[bout]][bee,];
                    if(seqBee[length(seqBee)]!=1) seqBee = c(seqBee[which(seqBee!=0)],1);
                    lengthSeq = length(seqBee);
                    seqBee = c(seqBee,rep(0,times=(maxLength+2)-lengthSeq));
                    matrixOfVisitationSequences[i,]=seqBee;
                  
                
                
                matrixOfVisitationSequences = cbind(rep(1:numberOfSimulations,each=numberOfBouts*numberOfBees),
                                                    rep(1:numberOfBouts,times=numberOfSimulations,each=numberOfBees),
                                                    rep(1:numberOfBees,times=numberOfSimulations*numberOfBouts),
                                                    matrixOfVisitationSequences);
                
                write.csv(matrixOfVisitationSequences,paste(outputFolderSim,"/matrixOfVisitationSequences.csv",sep=""),row.names=F);
                
                #### Transform the Route Quality & Foraged Resources into arrays (temporary, should use arrays from the beginning)
                arrayOfRouteQuality = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                arrayOfRouteQualityAbs = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                arrayOfForagedResources = array(data=0,dim=c(numberOfSimulations,numberOfBouts,numberOfBees));
                vectorOfRawRouteQuality = c();
                vectorOfRouteQuality = c();
                for(ind in 1:numberOfBees)
                
                  matrixOfRouteQualityAbs = matrix(subset(matrixOfBeeData[,5],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  
                  if(!is.null(optimalRouteQuality)) matrixOfRouteQuality = matrixOfRouteQualityAbs/optimalRouteQuality;
                  
                  arrayOfRouteQuality[,,ind] = matrixOfRouteQuality;
                  arrayOfRouteQualityAbs[,,ind] = matrixOfRouteQualityAbs;
                  vectorOfRawRouteQuality = c(vectorOfRawRouteQuality,as.vector(t(arrayOfRouteQualityAbs[,,ind])))
                  vectorOfRouteQuality = c(vectorOfRouteQuality,as.vector(t(arrayOfRouteQuality[,,ind])))
                  
                  matrixOfForagedResources = matrix(subset(matrixOfBeeData[,4],matrixOfBeeData[,3]==ind),nrow=numberOfSimulations,ncol=numberOfBouts,byrow=T);
                  arrayOfForagedResources[,,ind] = matrixOfForagedResources;
                
                
                
                
                routeQualityAbsDF = data.frame(Bee = rep(1:numberOfBees,each = numberOfSimulations*numberOfBouts),
                                               Simulation = rep(1:numberOfSimulations,times = numberOfBees,each = numberOfBouts),
                                               Bout = rep(1:numberOfBouts, times = numberOfSimulations*numberOfBees),
                                               rawQuality = vectorOfRawRouteQuality,
                                               relativeQuality = vectorOfRouteQuality);
                write.csv(routeQualityAbsDF,paste(outputFolderSim,"/routeQualityDF.csv",sep=""),row.names=F);
                
                
                # arrayOfMatrixDistance
                write.csv(arrayOfMatrixDistance,paste(outputFolderSim,"/arrayOfMatrixDistance.csv",sep=""),row.names = F);
                
                
                # 4.4 - Videos ------------------------------------------------------------------
                if(videoOutput)
                
                  # Create a directory for video outputs
                  outputFolderTracking = paste(outputFolderSim,"/Tracking",sep="");
                  dir.create(outputFolderTracking,showWarnings = F);
                  
                  CreateVideoOutput(arrayGeometry,matrixOfVisitationSequences,listOfResourcesTaken,beeData,framerate,simulationsToPrint,outputFolderTracking)
                
              # arrayNumber
            # param.abandonFactor
          # param.learningFactor
        # param.useRouteCompare
       #gamma
     #beta
   #alphaneg
 #alphapos

simEnd = Sys.time();
simDuration = difftime(simEnd,simStart,units="secs");
cat("Simulation completed in :",as.numeric(simDuration),"seconds \n")

"""