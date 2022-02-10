'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''
import os
current_working_directory = os.getcwd()

import sys
sys.path.append('Functions')

from parameters import *
import simulation_functions


array_info = {
"environment_type": environment_type, 
"environment_size": environment_size,
"number_of_flowers" :number_of_flowers,
"number_of_patches": number_of_patches,
"patchiness_index": patchiness_index,
"flowers_per_patch":flowers_per_patch
}

simulation_parameters = {
"use_route_compare":use_route_compare,
"use_online_reinforcement": use_online_reinforcement,
"use_Q_learning":use_Q_learning,
"initialize_Q_table":initialize_Q_table,
"cost_of_flying":cost_of_flying,
"use_dynamic_beta":use_dynamic_beta,
"use_delay_start":use_delay_start,
"starting_bout_for_naive":starting_bout_for_naive
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

simulation_functions.simulation(current_working_directory,experiment_name,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,dist_factor,number_of_bouts,number_of_simulations,silent_sim,sensitivity_analysis)