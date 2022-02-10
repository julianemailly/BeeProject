'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd 
#import tkinter as tk # use tkinter for te video output? or maybe pygame?

# Other Functions  --------------------------------------------------------------

def character_match(str_list,chr_searched) : 
  """
  Description: 
    The function returns a list chr_matched such as chr_matched[i]=True if chr_searched is found in the string str_list[i] and chr_matched[i]=False otherwise
  Inputs: 
    str_list: list of strings
    chr_searched: character/substring searched
  Outputs: 
    chr_matched: list of bools 
  """
  if len(str_list) == 0 : 
    return(False)
  else : 
    chr_matched = np.array([chr_searched in str_list[k] for k in range (len(str_list))])
    return(np.array(chr_matched))


def find_character_position(str_list,chr_searched) : 
	"""
	Decription: 
		Function that also looks for a subset of characters "chr_searched" in a character object "str_list", and returns the position of the first occurence of tis "chr_searched" subset in the object.
	Inputs: 
		str_list: list of strings
    	chr_searched: character/substring searched
	Outputs: 
		match_position: list of position of the matched characters
	"""
	if len(str_list)  == 0 :
		print("No character object to scan. Returning None")
		return(None)
	else : 
		match_position = []
		for string in (str_list) :
			if chr_searched in string : 
				match_position.append(string.index(chr_searched))
			else : 
				match_position.append(None)
		return(np.array(match_position))


def create_video_output(array_geometry, matrix_of_visitation_sequences,list_of_resources_taken,bee_data,frame_rate,simulations_to_print_output_folder_tracking):
	"""
	Description:
		Create a video output for a given simulation.
	Inputs:

	Outputs:
	"""
	return()
