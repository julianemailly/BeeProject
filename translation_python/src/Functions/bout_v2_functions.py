import numpy as np
import pandas as pd
import learning_functions
import geometry_functions
import time as t


def initialize_bee_route(number_of_bees) :
	return(np.full((number_of_bees,1),0))

def initialize_resources_on_flowers(number_of_flowers):
	resources_on_flowers = np.full(number_of_flowers,1)
	resources_on_flowers[0]=0 # nest
	return(resources_on_flowers)

def initialize_array_of_vector_used(number_of_bees,number_of_flowers) :
	return(np.zeros((number_of_flowers,number_of_flowers,number_of_bees)))

def initialize_count_transitions(number_of_bees,number_of_flowers) : 
	return(np.full((number_of_bees,number_of_flowers,number_of_flowers),0))

def which_bee_is_still_foraging(bee_data,number_of_bees):
	return(np.where(bee_data[:number_of_bees,1]==0)[0])

def get_available_destinations(bee_info,number_of_flowers,current_position,previous_position,bee,array_of_vector_used):

	omitted_destinations = [current_position] # At least the self-loop is forbidden.

	if not (bee_info["allow_nest_return"][bee]) : # If condition, omit the nest.
		omitted_destinations.append(0)

	if bee_info["forbid_reverse_vector"][bee] and previous_position is not None : # If condition, omit the previous visit.
		omitted_destinations.append(previous_position)

	if bee_info["leave_after_max_fail"][bee] : 
		omitted_destinations=np.concatenate((omitted_destinations,np.where(array_of_vector_used[current_position,:,bee]==bee_info["number_of_max_fails"][bee])[0]))

	omitted_destinations = np.unique(omitted_destinations) # Make sure each flower is mentioned only once.
	potential_destinations = np.array([flower for flower in range (number_of_flowers)])

	for omitted_destination in omitted_destinations : 
		potential_destinations = np.delete(potential_destinations, np.where(potential_destinations==omitted_destination))

	return(potential_destinations)

def choose_next_position(number_of_visits,bout,bee_data,bee_info,number_of_flowers,number_of_bees,bee_route,bees_still_foraging,current_positions,learning_array_list,array_of_vector_used,array_geometry,count_transitions):

	use_Q_learning = bee_info["use_Q_learning"][0]
	is_flowers_in_conflict = [False for flower in range (number_of_flowers)]
	next_positions = np.full((number_of_bees,1),-1)

	for bee in bees_still_foraging :

		# Get available destination for this bee
		current_position = current_positions[bee]

		if number_of_visits>0 : 
			previous_position = bee_route[bee,-2]
		else : 
			previous_position = None

		available_destinations = get_available_destinations(bee_info,number_of_flowers,current_position,previous_position,bee,array_of_vector_used)
		destination_is_available = [(flower in available_destinations) for flower in range (number_of_flowers)]

		# Get the probabilities
		if use_Q_learning : 
			use_dynamic_beta = bee_info["use_dynamic_beta"][0]
			if use_dynamic_beta : 
				beta = bee_info["beta_vector"][bee][bout]
			else : 
				beta = bee_info["beta"][bee]
			probabilities = learning_functions.softmax(learning_array_list[bee][current_position,destination_is_available],beta)
		else : 
			probabilities = learning_array_list[bee][current_position,destination_is_available]
			sum_of_probabilities = np.sum(probabilities)
			if sum_of_probabilities !=0 : 
				probabilities = probabilities/sum_of_probabilities

		# Choose the next destination
		

		if len(probabilities)==0 or np.sum(probabilities)==0: # The bee does not have any possibility
			if current_position!=0 : # The bee was not in the nest
				next_position = 0 # Go to the nest
			else : # The bee was already in the nest
				next_position = -1 # End of bout for this bee
		else : # There is at least one potential destination

			next_position = np.random.choice(a=available_destinations,p=probabilities)

		# Update the distance travelled in bee_data
		if next_position != -1 :
			current_position_coordinates = (array_geometry.loc[current_position,"x"],array_geometry.loc[current_position,"y"])
			next_position_coordinates = (array_geometry.loc[next_position,"x"],array_geometry.loc[next_position,"y"])
			bee_data[bee,2]+=geometry_functions.distance(current_position_coordinates,next_position_coordinates)

		# Update matrix of first transitions
		count_transitions[bee,current_position,next_position] += 1

		# Check if the next destination will trigger a conflict
		if (next_position != -1) and (next_position != 0) and (not is_flowers_in_conflict[next_position]) and (next_position in next_positions) : 
			is_flowers_in_conflict[next_position]=True

		# Update in the next_postions array
		next_positions[bee,0]=next_position

	bee_route = np.concatenate((bee_route,next_positions),axis=1)

	flowers_in_conflict = np.where(is_flowers_in_conflict)[0]
	return(flowers_in_conflict,bee_route)

def resolve_conflit(flowers_in_conflict,bee_route,number_of_bees):

	losers=[False for bee in range (number_of_bees)]
	destinations_chosen_by_each_bee = bee_route[:,-1].flatten()

	for flower in flowers_in_conflict : 
		individuals_in_competition = np.where(destinations_chosen_by_each_bee==flower)[0]
		
		winner = np.random.choice(individuals_in_competition)


		for bee in individuals_in_competition :
			if bee!=winner : 
				losers[bee]=True
	return(losers)

def punish_and_reward_bees(bee_route,bees_still_foraging,losers,resources_on_flowers,bee_info,bee_data,learning_array_list,array_geometry,count_transitions,array_of_vector_used):

	cost_of_flying = bee_info["cost_of_flying"][0]
	use_Q_learning = bee_info["use_Q_learning"][0]
	alpha_pos = bee_info["alpha_pos"]
	alpha_neg = bee_info["alpha_neg"]
	gamma = bee_info["gamma"]
	learning_factor = bee_info["learning_factor"]
	abandon_factor = bee_info["abandon_factor"]

	for bee in bees_still_foraging : 

		previous_flower,current_flower = bee_route[bee,-2],bee_route[bee,-1]
		if current_flower == -1 or count_transitions[bee,previous_flower,current_flower]>=2 : 
			reward = None # No learning in this case because the bee has finished its bout or this transition has already been learnt 

		else : 

			if losers[bee] : # If bee was kicked out from the flower, it is punished
				reward = 0
				array_of_vector_used[previous_flower,current_flower,bee]+=1

			else : 

				if resources_on_flowers[current_flower]!=0 : # The flower is full: the bee feeds and gets rewarded
					reward = 1
					resources_on_flowers[current_flower] -= 1
					bee_data[bee,0] +=1

				else : # The flower was empty: the bee is punished
					reward = 0
					array_of_vector_used[previous_flower,current_flower,bee]+=1

		# Apply online learning
		learning_functions.online_learning(bee,cost_of_flying,array_geometry,use_Q_learning,learning_array_list,previous_flower,current_flower,reward,alpha_pos[bee],alpha_neg[bee],gamma[bee],learning_factor[bee],abandon_factor[bee])

def check_if_bout_finished(bee_route,bees_still_foraging,bee_data):
	for bee in bees_still_foraging : 
		# Case 1: was in the nest and no possible destiation. In this case, bee_route[bee,-1] = -1
		# Case 2: the bee has reached the maximum travelled distance
		# Case 3: the crop of the bee is full
		# Case 4: allow_nest_return is true for this bee and it has chosen the nest
		if (bee_route[bee,-1] == -1) or (bee_data[bee, 2]>bee_info["max_distance_travelled"][bee]) or (bee_data[bee, 0]==bee_info["max_crop"][bee]) or (bee_info["allow_nest_return"][bee] and bee_route[bee,-1] == 0): 
			bee_data[bee, 1] = 1

def route_qualities_of_the_bout(optimal_route_quality_1_ind,optimal_route_quality_2_ind,bee_route,number_of_bees,bee_data) :
	route_qualities = []

	for bee in range(number_of_bees):

		route = bee_route[bee,:]
		route_quality = geometry_functions.compute_route_quality(bee_data[bee,0],bee_data[bee,2])
		route_quality=round(route_quality,8)
		route_qualities.append(route_quality)

		if optimal_route_quality_1_ind is not None and optimal_route_quality_1_ind<route_quality : 
			optimal_route_quality_1_ind = route_quality

	if optimal_route_quality_2_ind is not None: 
		group_route_quality = np.sum(route_qualities) 
		group_route_quality=round(group_route_quality,8)
		if optimal_route_quality_2_ind<group_route_quality: 
			optimal_route_quality_2_ind = group_route_quality

	return(route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind)


def simulate_bout(bout,bee_data,bee_info,learning_array_list,array_geometry,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_bees=None):

	beg = t.time()

	if number_of_bees == None :
		number_of_bees = np.shape(bee_data)[0]

	number_of_flowers = len(array_geometry.index)

	bee_route = initialize_bee_route(number_of_bees)
	resources_on_flowers = initialize_resources_on_flowers(number_of_flowers)
	array_of_vector_used = initialize_array_of_vector_used(number_of_bees,number_of_flowers) # only used if leave_after_max_fail
	count_transitions = initialize_count_transitions(number_of_bees,number_of_flowers)

	bees_still_foraging = which_bee_is_still_foraging(bee_data,number_of_bees)
	number_of_visits = 0

	while len(bees_still_foraging!=0): # At least one bee is still foraging

		current_positions = bee_route[:,-1]
		flowers_in_conflict,bee_route = choose_next_position(number_of_visits,bout,bee_data,bee_info,number_of_flowers,number_of_bees,bee_route,bees_still_foraging,current_positions,learning_array_list,array_of_vector_used,array_geometry,count_transitions)
		losers=resolve_conflit(flowers_in_conflict,bee_route,number_of_bees)
		punish_and_reward_bees(bee_route,bees_still_foraging,losers,resources_on_flowers,bee_info,bee_data,learning_array_list,array_geometry,count_transitions,array_of_vector_used)
		check_if_bout_finished(bee_route,bees_still_foraging,bee_data)

		bees_still_foraging = which_bee_is_still_foraging(bee_data,number_of_bees)
		number_of_visits += 1

	route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind=route_qualities_of_the_bout(optimal_route_quality_1_ind,optimal_route_quality_2_ind,bee_route,number_of_bees,bee_data)

	end = t.time()
	print(end-beg)

	return(bee_route,route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind) 



import management_of_data_functions
import environment_generation_functions

bout = 0
number_of_bees = 2

parameters_dict = {
"use_route_compare":False,
"use_online_reinforcement": True,
"use_Q_learning":False,
"initialize_Q_table": "distance",
"max_distance_travelled":3000,
"dist_factor" : 2,
"cost_of_flying":False,
"use_dynamic_beta":False,
"use_delay_start":False,
"starting_bout_for_naive":None,
"probability_of_winning": 0.5,
"max_distance_travelled": 3000,
"max_crop":5,
"beta_vector":None,
"leave_after_max_fail": False, 
"number_of_max_fails":2,
"forbid_reverse_vector": True,
"allow_nest_return": True,
"beta": 15,
"alpha_pos":0.4,
"alpha_neg":0.4,
"gamma":0.,
"learning_factor":1.5,
"abandon_factor":0.75
}

array_info = {
"environment_type": "generate", 
"environment_size": 500,
"number_of_flowers" :10,
"number_of_patches": 1,
"patchiness_index": 0,
"flowers_per_patch": None
}

array_number = 0
reuse_generated_arrays = True
current_working_directory = "C:/Users/julia/BeeProject/translation_python/src"
silent_sim = False


array_geometry, array_info, array_folder = environment_generation_functions.create_environment (array_info, array_number, reuse_generated_arrays,current_working_directory, silent_sim)


bee_data = management_of_data_functions.initialize_bee_data(number_of_bees)
bee_info = management_of_data_functions.initialize_bee_info(number_of_bees,parameters_dict)


if bee_info["use_Q_learning"][0] : 
	learning_array_list = management_of_data_functions.initialize_Q_table_list (bee_info["initialize_Q_table"][0], array_geometry, bee_info["dist_factor"][0], number_of_bees,bee_info["allow_nest_return"])
else : 
	learning_array_list = management_of_data_functions.initialize_probability_matrix_list(array_geometry,bee_info["dist_factor"][0],number_of_bees,bee_info["allow_nest_return"])

optimal_route_quality_1_ind = 0
optimal_route_quality_2_ind = 0

bee_route,route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind = simulate_bout(bout,bee_data,bee_info,learning_array_list,array_geometry,optimal_route_quality_1_ind,optimal_route_quality_2_ind)
