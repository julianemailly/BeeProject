'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

from parameters import *
import simulation_functions


array_info = {"environment_type": environment_type, "number_of_flowers" :number_of_flowers, "number_of_patches": number_of_patches, "patchiness_index": patchiness_index, "env_size": env_size, "flowers_per_patch":flowers_per_patch}

parameters_loop = {"initialize_Q_table":initialize_Q_table,"starting_bout_for_naive":starting_bout_for_naive,"different_experience_simulation":different_experience_simulation,"route_compare":use_route_compare,
"max_distance_travelled": max_distance_travelled,"max_crop":max_crop,"dynamic_beta":dynamic_beta,"beta_QL_vector":beta_QL_vector,"leave_after_max_fail": leave_after_max_fail, 
"max_fails":number_of_max_fails,"forbid_reverse_vector": forbid_reverse_vector,"allow_nest_return": allow_nest_return,"beta_QL": beta_QL,"alpha_pos":alpha_pos,"alpha_neg":alpha_neg,"gamma_QL":gamma_QL,"use_Q_learning":use_Q_learning,
"online_reinforcement": online_reinforcement,"cost_of_flying":cost_of_flying,"learning_factor":learning_factor,"abandon_factor":abandon_factor}

for key in parameters_loop : 
  if not isinstance(parameters_loop[key],list) : 
  	parameters_loop[key] = [parameters_loop[key]]

simulation_functions.simulation(test_name_general,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,dist_factor,number_of_bouts,number_of_simulations,silent_sim,sensitivity_analysis=True)