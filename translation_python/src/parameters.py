'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

# Parameters --------------------------------------------------------------

# Array details.
test_name_general = "Example" # An identification name for the simulation.
environment_type = "generate" # Either (i) input an array name (refer to folder names in the "Arrays" folder - Note that this folder is automatically created when you first generate an array), or (ii) input "generate" to generate procedural arrays. For the latter, provide details in 3.1.1

## Details for procedural arrays (if  using "generate" as environmentType). If not used, skip to the next paragraph.
number_of_flowers = 6 # Number of flowers (all patches combined)
number_of_patches = 1 # Number of patches in which the flowers are distributed
patchiness_index = 0 # Patchiness of the array. Takes values between 0 (homogeneous) and 1 (very heterogeneous).
env_size = 500 # Size of the environment in meters. 
flowers_per_patch = None # Number of flowers per patch. If only one patch, set to None. Takes one value per patch, sum need to be equal to numberOfResources.
number_of_arrays = 1 # Number of different arrays created using the values above. Only used if environmentType == "generate".
reuse_generated_arrays =  True # If  True and there already are generated arrays with the same parameters, they will be used instead of generating new ones.

# Simulation parameters
number_of_bees = 2 # Number of bees moving simultaneously in the environment.
number_of_simulations = 5 # Number of simulations for each set of parameter.
number_of_bouts = 10 # Number of bouts in each simulation.
dist_factor = 2 # This variable contains the power at which the distance is taken in the [probability = 1/d^distFactor] function to estimate movement probabilities.
max_crop = 5

# Forager parameters. 
# Each item of the list objects will replicate the simulations for its values. The model will run for all combinations of parameters. May quickly increase computation time.
use_route_compare = False #  True : Use the route-based learning algorithm (Reynolds et al., 2013). False : Use our vector-based algorithm.
# /!\ Warning : The route-based algorithm is not identical to that of Reynolds et al., 2013. Mainly, it uses the route with revisits when assessing route quality.
learning_factor = 1. # Strength of the learning process. Translates as a multiplication of any vector probability P by this factor (Should be 1 <= learningFactor)
abandon_factor = 0.75 # Strength of the abandon process. Translates as a multiplication of any vector probability P by this factor (Should be 0 <= abandonFactor <= 1)
max_distance_travelled = 3000 # Maximum distance the bee can travel before being exhausted. After reaching this threshold, the bee goes back to its nest no matter what.

# Output management & technical parameters.
video_output = False # If  True, outputs videos showing the bee movements during simulation. /!\ Requires FFMPEG installed on computer /!\ You can find FFMPEG at the following address : https://ffmpeg.org/download.html. Also, not tested on other OS than Windows 10, may fail to work on Mac.
framerate = 8  # The number of frames per seconds on the video output
simulations_to_print = 1 # Select the number of videos to print. Must not be above numberOfSimulations.
silent_sim = False  # If False, provides comments in the Console on the progression of the simulation.

# Advanced parameters
leave_after_max_fail = False  # If a vector use leads to no reward number_of_max_fails times in a bout, forbid its use until the end of the bout.
number_of_max_fails = 2  # If leaveAfterMaxFail is  True, set the number of fails needed for the bee to ignore a vector.
allow_nest_return =  True  # If  True, probabilities to go to the nest from any position are not set to 0. If the nest is reached, the bout ends.
forbid_reverse_vector =  True  # If  True, foragers ignore reverse vectors. Example : if it just did 2->3, the prob of doing 3->2 is 0 until its next movement.
different_experience_simulation = False  # If  True & numberOfBees = 2 : Setup the simulation so that Bee 1 starts foraging for n bouts before Bee 2 starts foraging.
starting_bout_for_naive = None # Bout at which each bee starts foraging. Should be < to numberOfBouts. Expects as many values as the numberOfBees if differentExperienceSimulation is  True.
online_reinforcement =  True  # If  True, probability changes after a good/bad experience is immediate and not at the end of the bout.

# Parameters for the Q-learning/Rescorla-Wagner model
use_Q_learning = True  # True: if you want to use this Q learning model, False: if you want to use T. Dubois' model
if (use_Q_learning) : 
	online_reinforcement = True # Q-Learning is necessarily online
initialize_Q_table = "distance"  # 'zero' if yo want the Q table to be initialized as a None matrix, 'distance' if you want it to be initialized as the 1/d^distFactor matrix, 'noisydist if you want to add noise to the distance ditribution
alpha_pos = [0.4]  # Positive reinforcement learning rate: 0<=alphaPos<=1
alpha_neg=0.4  # Negative reinforcement learning rate: 0<=alphaNeg<=1
beta= 15  # Exploration-exploitation parameter: 0<=beta
gamma = 0  # Temporal discounting factor: 0<=gamma<=1. Here, set to 0 for simplicity
dynamic_beta=False # Use a dynamic beta that changes throughout time
beta_vector = None
if (dynamic_beta) : 
	starting_beta = 1 # Beta at the beginning of the simulation
	final_beta = 10 # Beta at the end of the simulation
	bouts_to_increase_beta = 20 # Number of bouts during which beta increases
	beta_vector = [final_beta for k in range (number_of_bouts)]
	for bout in range (bouts_to_increase_beta) : 
		beta_vector[bout] = starting_beta + bout*(final_beta-starting_beta +1)/bouts_to_increase_beta
cost_of_flying = False # True: will define the reward as presence_of_nectar_in_tne_flower - travelled_distance_from_the_previous_flower / max_distance_between_flowers

sensitivity_analysis = True