'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Parameters --------------------------------------------------------------


########## CAREFUL: HOW TO INITIALIZE THE PARAMETERS ##########
'''
This code allows for testing different combinations of parameters in different simulations.

The genral architecture of each paarmeter is either:
	1. parameter = value, and this value will be used for each bee
	2. parameter = [value_for_test_1, ..., value_for_test_m], and m values will be tested in seperate simulations, each bee having the same value for this parameter. Please note that parameter = value and parameter = [value] is equivalent
	3. parameter = [list_of_value_for_each_bee_for_test_1, ..., list_of_value_for_each_bee_for_test_m], which is the same as in case 2 but here, each bee can have different values. The list of different values for eahc bee must IMPERATIVELY have the same length as number_of_bees.
Note that you can combine case 2 and 3 by specifying either a list or a single value for each test.

Let us take the example of alpha_pos, which is used during Q-learning to update the Q tables in case of a positive outcome.
	1. The parameter can take a single value: e.g. alpha_pos = 0.5
	2. The parameter can take a list of values: e.g. alpha_pos = [0.2, 0.5, 0.7]
	3. The parameter can take a list of lists of values: e.g alpha_pos = [[0.2,0.5], [0.5, 0.2], [0.7, 0.5]]

Remark: if you want to test 1 set of values for different bees (only 1 test but specify the value for each bee), do your_parameter = [your_list_of_values]. For example, alpha_pos = [[0.2,0.5]]

Please be careful when initializing parameters that are necessarily lists. Please find the instructions to initialize those specific parameters just below: 
	beta_vector (when use_dynamic_beta is True): this parameter always take a list of beta values for each bout. Therefore, to do only one test (precise the dynamic values of beta for each bee for one test) you must necessarily implement it such as beta_vector = [[list_of_value_per_bout_bee_1, ...,list_of_value_per_bout_bee_n]] 
'''




########## ARRAY INFORMATION ##########
'''
These parameters can only take a single value (not a list).
'''

# Array details.
experiment_name = "Example" # An identification name for the experiment.
environment_type = "generate" # Either (i) input an array name (refer to folder names in the "Arrays" folder - Note that this folder is automatically created when you first generate an array), or (ii) input "generate" to generate procedural arrays. For the latter, provide details in 3.1.1

# Details for procedural arrays (if  using "generate" as environmentType). If not used, skip to the next paragraph.
number_of_flowers = 10 # Number of flowers (all patches combined)
number_of_patches = 1 # Number of patches in which the flowers are distributed
patchiness_index = 0 # Patchiness of the array. Takes values between 0 (homogeneous) and 1 (very heterogeneous).
environment_size = 500 # Size of the environment in meters. 
flowers_per_patch = None # Number of flowers per patch. If only one patch, set to None. Takes one value per patch, sum need to be equal to numberOfResources.
number_of_arrays = 1 # Number of different arrays created using the values above. Only used if environmentType == "generate".
reuse_generated_arrays =  True # If  True and there already are generated arrays with the same parameters, they will be used instead of generating new ones.





########## PARAMETERS OF THE EXPERIMENT ##########
'''
These parameters can only take a single value (not a list).
'''

# Simulation parameters
number_of_simulations = 100 # Number of simulations for each set of parameter.
number_of_bouts = 30 # Number of bouts in each simulation.
number_of_bees = 2 # Number of bees moving simultaneously in the environment.
dist_factor = 2 # This variable contains the power at which the distance is taken in the [probability = 1/d^distFactor] function to estimate movement probabilities when not use_Q_learning or if initialize_Q_table = "distance".

# Technical parameters
silent_sim = False  # If False, provides comments in the Console on the progression of the simulation.
sensitivity_analysis = False # If True, computes the absolute difference between successive learning arrays for each bee
video_output = False # If  True, outputs videos showing the bee movements during simulation. NOT IMPLEMENTED YET.
stochasticity = True # Can deactivate the stochasticity in the simulation (useful to debug the code)




########## PARAMETERS OF THE SIMULATION ##########
'''
These parameters can be initialized as a list but each bee much have the same value for each parameter (see case 2 in the initilalization guide at line 9 fo this document).
'''

# Algorithm specifications
use_Q_learning = True  # True: if you want to use this Q learning model, False: if you want to use T. Dubois' model
initialize_Q_table = "distance"  # 'zero' if yo want the Q table to be initialized as a None matrix, 'distance' if you want it to be initialized as the 1/d^distFactor matrix, 'noisydist if you want to add noise to the distance ditribution
cost_of_flying = False # True: will define the reward as presence_of_nectar_in_tne_flower - travelled_distance_from_the_previous_flower / max_distance_between_flowers
use_dynamic_beta=False # Use a dynamic beta that changes throughout time





########## PARAMETERS OF THE BEES ##########
'''
These parameters can take different values for each test and each bee. 
EXCEPTION: beta_vector (see line 27)
'''

# Forager parameters
max_distance_travelled = 3000 # Maximum distance the bee can travel before being exhausted. After reaching this threshold, the bee goes back to its nest no matter what.
max_crop = 5 # Maximum number of flowers for the bee's crop to be full

# Parameters controlling the available destinations when choosing a new flower
leave_after_max_fail = False  # If a vector use leads to no reward number_of_max_fails times in a bout, forbid its use until the end of the bout.
number_of_max_fails = 2  # If leave_after_max_fail is  True, set the number of fails needed for the bee to ignore a vector.
allow_nest_return =  True  # If  True, probabilities to go to the nest from any position are not set to 0. If the nest is reached, the bout ends.
forbid_reverse_vector =  True  # If  True, foragers ignore reverse vectors. Example : if it just did 2->3, the prob of doing 3->2 is 0 until its next movement.

# Learning parameters
learning_factor = 1.5 # Strength of the learning process. Translates as a multiplication of any vector probability P by this factor (Should be 1 <= learningFactor)
abandon_factor = 0.75 # Strength of the abandon process. Translates as a multiplication of any vector probability P by this factor (Should be 0 <= abandonFactor <= 1)
alpha_pos = 0.4  # Positive reinforcement learning rate: 0<=alphaPos<=1
alpha_neg=0.4  # Negative reinforcement learning rate: 0<=alphaNeg<=1
beta= 15.  # Exploration-exploitation parameter: 0<=beta
gamma = 0  # Temporal discounting factor: 0<=gamma<=1. Here, set to 0 for simplicity
beta_vector = None
if (use_dynamic_beta) : # One possible way to implement beta_vector
	starting_beta = 1 # Beta at the beginning of the simulation
	final_beta = 10 # Beta at the end of the simulation
	bouts_to_increase_beta = 20 # Number of bouts during which beta increases
	beta_values_in_bout = [final_beta for k in range (number_of_bouts)] # Creating the list of beta values for each bout
	for bout in range (bouts_to_increase_beta) : 
		beta_values_in_bout[bout] = starting_beta + bout*(final_beta-starting_beta +1)/bouts_to_increase_beta
	beta_vector = [] # Creating the beta_vector
	for bee in range (number_of_bees): 
		beta_vector.append(beta_values_in_bout)
	beta_vector = [beta_vector] # Now has this structure: beta_vector = [[list_of_value_per_bout_bee_1, ...,list_of_value_per_bout_bee_n]]





########## INITIALIZING THE PARAMETERS DICTIONARIES ##########

array_info = {
"environment_type": environment_type, 
"environment_size": environment_size,
"number_of_flowers" :number_of_flowers,
"number_of_patches": number_of_patches,
"patchiness_index": patchiness_index,
"flowers_per_patch":flowers_per_patch
}

simulation_parameters = {
"use_Q_learning":use_Q_learning,
"initialize_Q_table":initialize_Q_table,
"dist_factor" : dist_factor,
"cost_of_flying":cost_of_flying,
"use_dynamic_beta":use_dynamic_beta
}

parameters_of_individuals  = {
"max_distance_travelled": max_distance_travelled,
"max_crop":max_crop,
"beta_vector":beta_vector,
"leave_after_max_fail": leave_after_max_fail, 
"number_of_max_fails":number_of_max_fails,
"forbid_reverse_vector": forbid_reverse_vector,
"allow_nest_return": allow_nest_return,
"beta": beta,
"alpha_pos":alpha_pos,
"alpha_neg":alpha_neg,
"gamma":gamma,
"learning_factor":learning_factor,
"abandon_factor":abandon_factor
}

parameters_loop = {}
parameters_loop.update(parameters_of_individuals)
parameters_loop.update(simulation_parameters)

for key in parameters_loop : 
  if not isinstance(parameters_loop[key],list) : 
  	parameters_loop[key] = [parameters_loop[key]]