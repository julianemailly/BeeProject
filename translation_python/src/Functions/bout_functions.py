import numpy as np
import pandas as pd
import learning_functions
import geometry_functions
import management_of_data_functions


def get_available_destinations(bee_info,number_of_flowers,current_position,previous_position,bee,count_failed_transitions):
	"""
	Description:
	Inputs:
		bee_info: dataframe of parameters for each bee
		number_of_flowers: number of flowers including nest
		current_position: index of the current position of the bee (flower ID)
		previous_position: index of the previous position of the bee (flower ID)
		bee: bee ID
		count_failed_transitions: matrix of size (numer_of_bees, number_of_flowers,number_of_flowers) counting transitions that led to a negative outcome
	Outputs: 
		available_destinations: list of available destinations
	"""
	omitted_destinations = [current_position] # At least the self-loop is forbidden.

	if not (bee_info["allow_nest_return"][bee]) : # If condition, omit the nest.
		omitted_destinations.append(0)

	if bee_info["forbid_reverse_vector"][bee] and previous_position is not None : # If condition, omit the previous visit.
		omitted_destinations.append(previous_position)

	if bee_info["leave_after_max_fail"][bee] : 
		omitted_destinations=np.concatenate((omitted_destinations,np.where(count_failed_transitions[bee,current_position,:]==bee_info["number_of_max_fails"][bee])[0]))

	omitted_destinations = np.unique(omitted_destinations) # Make sure each flower is mentioned only once.
	available_destinations = np.array([flower for flower in range (number_of_flowers)])

	for omitted_destination in omitted_destinations : 
		available_destinations = np.delete(available_destinations, np.where(available_destinations==omitted_destination))

	return(available_destinations)

def choose_next_position(distance_travelled,number_of_visits,bout,bee_info,number_of_flowers,number_of_bees,bee_route,is_bee_still_foraging,is_bee_going_back_to_nest,is_bee_in_nest,learning_array_list,count_failed_transitions,array_geometry,count_transitions,stochasticity):
	"""
	Description:
		Choose the next position of each bee still foraging and ensures that the bees that have finished their bout are going back to the nest
	Inputs:
		distance_travelled: list of distance travelled so far by each bee
		number_of_visits: number of visits made during this bout so far
		bout: index of bout
		bee_info: dataframe of parameters for each bee
		number_of_flowers: number of flowers including nest
		number_of_bees: number of bees 
		bee_route: matrix of size number_of_bees*number_of_visits(so far) that contain the current sequence of actions for each bee
		is_bee_still_foraging: bool list of size number_of_bees such as is_bee_still_foraging[bee] is True if bee is still foraging
		is_bee_going_back_to_nest: bool list of size number_of_bees such as is_bee_going_back_to_nest[bee] is True if bee is going back to the nest
		is_bee_in_nest: bool list of size number_of_bees such as is_bee_in_nest[bee] is True if bee is in the nest
		current_positions: list of current positions of the bees
		learning_array_list: numpy array of size number_of_bees*number_of_flowers*number_of_flowers giving the learning arry of each bee
		count_failed_transitions: matrix of size (numer_of_bees, number_of_flowers,number_of_flowers) counting transitions that led to a negative outcome
		array_geometry: pd dataframe storing the position fo each flower (including nest)
		count_transitions: matrix of size (numer_of_bees, number_of_flowers,number_of_flowers) counting the number of transitions
		stochasticity: bool, if False deactivate stochasticity in the simulation
	Outputs: 
		(some lists are updated in place)
		flowers_in_conflict: list of flowers that were chosen by two or more bees
		bee_route: updated bee_route with the new positions
	"""

	use_Q_learning = bee_info["use_Q_learning"][0]
	is_flowers_in_conflict = [False for flower in range (number_of_flowers)]
	next_positions = np.full((number_of_bees,1),-1)

	current_positions = bee_route[:,-1]

	for bee in range (number_of_bees) :

		# Ensures that the bees that have finished their bout are going back to the nest
		if is_bee_going_back_to_nest[bee] : 

			current_position = current_positions[bee]

			if not(current_position==0 or current_position==-1) : # If bee is not already in nest, make it go back and compute the associated distance

				next_position = 0 # Go back to nest
				next_positions[bee,0] = next_position # Update the vector of next positions

				current_position_coordinates = (array_geometry.loc[current_position,"x"],array_geometry.loc[current_position,"y"]) # Compute the coordinates of current position
				next_position_coordinates = (0.,0.) # Coordinates of the nest
				distance_travelled[bee] += geometry_functions.distance(current_position_coordinates,next_position_coordinates)

			is_bee_going_back_to_nest[bee] = False
			is_bee_in_nest[bee]=True

		if is_bee_still_foraging[bee] :

			# Get available destination for this bee
			current_position = current_positions[bee]

			if number_of_visits>0 : 
				previous_position = bee_route[bee,-2]
			else : 
				previous_position = None

			available_destinations = get_available_destinations(bee_info,number_of_flowers,current_position,previous_position,bee,count_failed_transitions)
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

				if stochasticity : 
					next_position = np.random.choice(a=available_destinations,p=probabilities)
				else : 
					next_position = available_destinations[np.argmax(probabilities)]

			# Update the distance travelled
			if next_position != -1 :
				current_position_coordinates = (array_geometry.loc[current_position,"x"],array_geometry.loc[current_position,"y"])
				next_position_coordinates = (array_geometry.loc[next_position,"x"],array_geometry.loc[next_position,"y"])
				distance_travelled[bee] +=geometry_functions.distance(current_position_coordinates,next_position_coordinates)

			# Update matrix of first transitions
			if next_position != -1 :
				count_transitions[bee,current_position,next_position] += 1

			# Check if the next destination will trigger a conflict
			if (next_position != -1) and (next_position != 0) and (not is_flowers_in_conflict[next_position]) and (next_position in next_positions) : 
				is_flowers_in_conflict[next_position]=True

			# Update in the next_postions array
			next_positions[bee,0]=next_position

	bee_route = np.concatenate((bee_route,next_positions),axis=1)

	flowers_in_conflict = np.where(is_flowers_in_conflict)[0]
	return(flowers_in_conflict,bee_route)

def resolve_conflit(flowers_in_conflict,bee_route,number_of_bees,stochasticity):
	"""
	Description:
		Solve the competitive interaction on the relevant flower
	Inputs:
		flowers_in_conflict: list of flowers that were chosen by two or more bees
		bee_route: updated bee_route with the new positions
		number_of_bees: number of bees
		stochasticity: bool, if False deactivate stochasticity in the simulation
	Outputs: 
		losers: individuals that have lost the competition
	"""
	losers=[False for bee in range (number_of_bees)]
	destinations_chosen_by_each_bee = bee_route[:,-1].flatten()

	for flower in flowers_in_conflict : 
		individuals_in_competition = np.where(destinations_chosen_by_each_bee==flower)[0]
		
		if stochasticity : 
			winner = np.random.choice(individuals_in_competition)
		else :
			winner = 0


		for bee in individuals_in_competition :
			if bee!=winner : 
				losers[bee]=True
	return(losers)

def punish_and_reward_bees(number_of_resources_foraged,number_of_bees,bee_route,is_bee_still_foraging,losers,resources_on_flowers,bee_info,learning_array_list,array_geometry,count_transitions,count_failed_transitions):
	"""
	Description:
		Apply online learning for each bee still foraging
	Inputs:
		number_of_resources_foraged: list of resources foraged so far by each bee
		number_of_bees: number_of_bees
		bee_route: matrix of size number_of_bees*number_of_visits(so far) that contain the current sequence of actions for each bee
		is_bee_still_foraging: bool list of size number_of_bees such as is_bee_still_foraging[bee] is True if bee is still foraging
		losers: individuals that have lost the competition
		resources_on_flowers: list of resources on flowers
		bee_info: dataframe of parameters for each bee
		learning_array_list: numpy array of size number_of_bees*number_of_flowers*number_of_flowers giving the learning arry of each bee
		array_geometry: pd dataframe storing the position fo each flower (including nest)
		count_transitions: matrix of size (numer_of_bees, number_of_flowers,number_of_flowers) counting the number of transitions
		count_failed_transitions: matrix of size (numer_of_bees, number_of_flowers,number_of_flowers) counting transitions that led to a negative outcome
	Outputs: 
		learning array_list is modified in place
	"""
	cost_of_flying = bee_info["cost_of_flying"][0]
	use_Q_learning = bee_info["use_Q_learning"][0]
	alpha_pos = bee_info["alpha_pos"]
	alpha_neg = bee_info["alpha_neg"]
	gamma = bee_info["gamma"]
	learning_factor = bee_info["learning_factor"]
	abandon_factor = bee_info["abandon_factor"]

	for bee in range (number_of_bees) : 

		if is_bee_still_foraging[bee] : 

			previous_flower,current_flower = bee_route[bee,-2],bee_route[bee,-1]

			if current_flower == -1 or current_flower == 0 or count_transitions[bee,previous_flower,current_flower]>=2 : 
				reward = None # No learning in this case because the bee has finished its bout or this transition has already been learnt 
			else : 

				if losers[bee] : # If bee was kicked out from the flower, it is punished
					reward = 0
					count_failed_transitions[bee,previous_flower,current_flower]+=1

				else : 

					if resources_on_flowers[current_flower]!=0 : # The flower is full: the bee feeds and gets rewarded
						reward = 1
						resources_on_flowers[current_flower] -= 1
						number_of_resources_foraged[bee] +=1

					else : # The flower was empty: the bee is punished
						reward = 0
						count_failed_transitions[bee,previous_flower,current_flower]+=1

			# Apply online learning
			learning_functions.apply_online_learning(bee,cost_of_flying,array_geometry,use_Q_learning,learning_array_list,previous_flower,current_flower,reward,alpha_pos[bee],alpha_neg[bee],gamma[bee],learning_factor[bee],abandon_factor[bee])

def check_if_bout_finished(number_of_resources_foraged,distance_travelled,number_of_bees,bee_route,is_bee_still_foraging,is_bee_going_back_to_nest,is_bee_in_nest,bee_info):
	"""
	Description:
		Check if the bout is finished for some bees
	Inputs:
		number_of_resources_foraged: list of resources foraged so far by each bee
		distance_travelled: list of distances travelled so far by each bee
		number_of_bees: number of bees
		bee_route: matrix of size number_of_bees*number_of_visits(so far) that contain the current sequence of actions for each bee
		is_bee_still_foraging: bool list of size number_of_bees such as is_bee_still_foraging[bee] is True if bee is still foraging
		is_bee_going_back_to_nest: bool list of size number_of_bees such as is_bee_going_back_to_nest[bee] is True if bee is going back to the nest
		is_bee_in_nest: bool list of size number_of_bees such as is_bee_in_nest[bee] is True if bee is in the nest
	Outputs: 
		Some lists are modified in place
	"""
	for bee in range(number_of_bees) : 
		if is_bee_still_foraging[bee] : 
			# Case 1: was in the nest and no possible destiation. In this case, bee_route[bee,-1] = -1 -> the bout is over and the bee is already in nest
			if (bee_route[bee,-1] == -1):
				is_bee_still_foraging[bee]=False
				is_bee_in_nest[bee]=True
			# Case 2: the bee has reached the maximum travelled distance
			# Case 3: the crop of the bee is full
			# Case 4: allow_nest_return is true for this bee and it has chosen the nest
			if (distance_travelled[bee]>bee_info["max_distance_travelled"][bee]) or (number_of_resources_foraged[bee]==bee_info["max_crop"][bee]) or (bee_info["allow_nest_return"][bee] and bee_route[bee,-1] == 0): 
				is_bee_still_foraging[bee]=False
				is_bee_going_back_to_nest[bee]=True

def route_qualities_of_the_bout(number_of_resources_foraged,distance_travelled,optimal_route_quality_1_ind,optimal_route_quality_2_ind,bee_route,number_of_bees) :
	"""
	Description:
		Computes the route qualities at the end of each bout and updates the optimal route quality known so far
	Inputs:
		number_of_resources_foraged list of resources foraged so far by each bee
		distance_travelled: list of distances travelled so far by each bee
		optimal_route_quality_1/2_ind: current approxiamtion of the optimal route quality for 1/2 bees
		bee_route: matrix of size number_of_bees*number_of_visits(so far) that contain the current sequence of actions for each bee
		number_of_bees: number_of_bees
	Outputs: 
		route_qualities: route quality for each bee
		optimal_route_quality_1/2_ind (updated)
	"""
	route_qualities = []

	for bee in range(number_of_bees):
		route = bee_route[bee,:]
		route_quality = geometry_functions.compute_route_quality(number_of_resources_foraged[bee],distance_travelled[bee])
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


def simulate_bout(bout,bee_info,learning_array_list,array_geometry,optimal_route_quality_1_ind,optimal_route_quality_2_ind,stochasticity,number_of_bees=None):
	"""
	Description:
		Simulate a bout
	Inputs:
		bout: index of bout
		bee_info: dataframe of parameters for each bee
		learning_array_list: numpy array of size number_of_bees*number_of_flowers*number_of_flowers giving the learning arry of each bee
		array_geometry: pd dataframe storing the position fo each flower (including nest)
		optimal_route_quality_1/2_ind: current approxiamtion of the optimal route quality for 1/2 bees
		stochasticity: if False, deactivate stochasticity
		number_of_bees: number_of_bees. If none, will take the number of rows in bee_info
	Outputs: 
		bee_route: route of each bee (1 row = 1 bee route)
		route_qualities: route quality for each bee
		optimal_route_quality_1/2_ind (updated)
		number_of_resources_foraged: number of resources foraged by each bee
	"""

	if number_of_bees == None :
		number_of_bees = len(bee_info.index)

	number_of_flowers = len(array_geometry.index)

	bee_route,resources_on_flowers,count_failed_transitions,count_transitions,is_bee_still_foraging,is_bee_going_back_to_nest,is_bee_in_nest,number_of_resources_foraged,distance_travelled = management_of_data_functions.initialize_bout_data(number_of_bees,number_of_flowers)

	number_of_visits = 0

	while np.sum(is_bee_still_foraging)!=0 or np.sum(is_bee_going_back_to_nest)!=0  : # At least one bee is still foraging or has not come back to the nest


		flowers_in_conflict,bee_route = choose_next_position(distance_travelled,number_of_visits,bout,bee_info,number_of_flowers,number_of_bees,bee_route,is_bee_still_foraging,is_bee_going_back_to_nest,is_bee_in_nest,learning_array_list,count_failed_transitions,array_geometry,count_transitions,stochasticity)
		losers=resolve_conflit(flowers_in_conflict,bee_route,number_of_bees,stochasticity)
		punish_and_reward_bees(number_of_resources_foraged,number_of_bees,bee_route,is_bee_still_foraging,losers,resources_on_flowers,bee_info,learning_array_list,array_geometry,count_transitions,count_failed_transitions)
		check_if_bout_finished(number_of_resources_foraged,distance_travelled,number_of_bees,bee_route,is_bee_still_foraging,is_bee_going_back_to_nest,is_bee_in_nest,bee_info)

		number_of_visits += 1

	route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind=route_qualities_of_the_bout(number_of_resources_foraged,distance_travelled,optimal_route_quality_1_ind,optimal_route_quality_2_ind,bee_route,number_of_bees)
	return(bee_route,route_qualities,optimal_route_quality_1_ind,optimal_route_quality_2_ind,number_of_resources_foraged) 