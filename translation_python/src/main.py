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


simulation_functions.simulation(current_working_directory,experiment_name,array_info,number_of_arrays,parameters_loop,number_of_bees, reuse_generated_arrays,dist_factor,number_of_bouts,number_of_simulations,silent_sim,sensitivity_analysis)