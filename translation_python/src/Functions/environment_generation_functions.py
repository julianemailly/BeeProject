'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import os
import numpy as np
import pandas as pd
import re
import copy
import geometry_functions
import other_functions

# Spatial Array Generation & Manipulation Functions --------------------------------------------------------------

def normalize_array_geometry(array_geometry) : 
  '''
  Description: 
    Set the first line (nest) at coordinates (0,0) and adjust all other coordinates.
  Input: 
    array_geometry: pandas dataframe with 4 columns: flower ID, x, y, patch ID
  Ouput: 
    array_geometry
  '''
  array_geometry["x"] = array_geometry["x"] - array_geometry["x"][0]
  array_geometry["y"] = array_geometry["y"] - array_geometry["y"][0]
  return(array_geometry)


def load_and_normalize_array_geometry(file_name) : 
  return(normalize_array_geometry(pd.read_csv(file_name)))


def create_array_ID(array_info,array_number) : 
  """
  Description:
    Create an unique name for an array
  Inputs:
    array_info: dictionary with the different characteristics of an array
    array_number: integer, gives the array number
  Outputs:
    array_ID: name of the array
  """
  if array_info['number_of_patches'] == 1 or array_info['flowers_per_patch'] is None: 
    flowers_per_patch_string = 'None'
  else : 
    flowers_per_patch_string = ''.join(map(str,array_info['flowers_per_patch']))
  array_ID = "Array-" + str(array_info["number_of_flowers"]) + "-"+ str(array_info["number_of_patches"]) + "-" + str(array_info["patchiness_index"]) + "-" + flowers_per_patch_string + "-" + str(array_info["environment_size"]) + "_" + "{:02d}".format(array_number)
  return(array_ID)


def get_ID_number_for_array(list_of_known_arrays,array_type_ID):
  """
  Description:
    Scan among known arrays to find a array ID not taken to give to the new array
  Inputs:
    list_of_known_arrays: list of the names of known arrays
    array_type_ID: ID of array type of the new array
  Outputs:
    array_ID_number: available ID number
  """
  indices_of_similar_arrays = np.where(other_functions.character_match(list_of_known_arrays,array_type_ID))[0]
  list_of_similar_arrays = list_of_known_arrays[indices_of_similar_arrays]
  number_of_similar_arrays = len(list_of_similar_arrays)
  available_numbers = [1 for i in range (number_of_similar_arrays+1)] # available_number[number] = 1 if number is abailable and 0 otherwise
  if number_of_similar_arrays > 0 :
    for i in range (number_of_similar_arrays) : 
      name_of_scanned_array = list_of_similar_arrays[i] 
      end_of_scanned_array = re.findall('\_0*\d*',name_of_scanned_array)[0]
      scanned_array_number = int(end_of_scanned_array[1:])
      available_numbers[scanned_array_number] = 0 # this number is not available
  array_ID_number = np.where(np.array(available_numbers)==1)[0][0]  # smallest number such as available_numbers[number] == 1 (and add 1 because we want the ID to start from 1)
  return(array_ID_number)


def generate_cartesian_coordinates_for_flower(min_distance,max_distance) :  # can be used for path centers
  """
  Description: 
    Generate cartesian coordinates for a flower or the center of a patch
  Inputs: 
    min_distance between a flower and a reference
    max_distance between a flower and a reference
  Outputs:
    Coordinates of the flower/patch
  """
  distance = np.random.uniform(low=min_distance,high=max_distance)
  azimuth = np.random.uniform(low=0,high=2*np.pi)
  return(geometry_functions.convert_polar_to_cartesian((distance,azimuth)))


def check_if_flower_is_sufficiently_far(coordinates_of_current_flower, minimal_distance_between_flowers ,coordinates_of_other_flowers) : # can be used for path centers
  """
  Description:
    Check if a flower/patch is at a sufficient distance from orther flowers/patches
  Inputs:
    coordinates_of_current_flower: coordinates of the flower that is examined
    minimal_distance_between_flowers: criterion for the minimal distance between 2 flowers/patches
    coordinates_of_other_flowers: matrix of size number_of_flowers_to_check x 2 giving the coordinates of the other flowers to be compared with the current flower
  Outputs:
    Boolean equal to the proposal "The flower/patch is sufficiently far"
  """
  number_of_flowers_to_check, _ = np.shape(coordinates_of_other_flowers)

  checking_distance_with_flower_number = 0
  distance_with_current_flower = geometry_functions.distance(coordinates_of_current_flower,coordinates_of_other_flowers[checking_distance_with_flower_number,:])

  while (checking_distance_with_flower_number<number_of_flowers_to_check-1) and (distance_with_current_flower>=minimal_distance_between_flowers) : 
    checking_distance_with_flower_number += 1
    distance_with_current_flower = geometry_functions.distance(coordinates_of_current_flower,coordinates_of_other_flowers[checking_distance_with_flower_number,:])

  return (checking_distance_with_flower_number == (number_of_flowers_to_check - 1) and distance_with_current_flower>=minimal_distance_between_flowers)


def generate_number_of_flowers_per_patch(number_of_flowers,number_of_patches) : 
  """
  Description:
    Generate a certain number of flowers per patch respecting the total number of flowers by sampling a Poisson distribution 
    More precisely, there is at least 1 flower per patch and the rest is sampled randomly
  Inputs:
    number_of_flowers: total number of flowers
    number_of_patches: total number of patches
  Outputs:
    Array with number of flowers per patch 
  """
  flowers_to_sample = number_of_flowers - number_of_patches
  lambda_poisson = flowers_to_sample/number_of_patches 
  sampled_flowers = np.zeros(number_of_patches)

  while(np.sum(sampled_flowers) != flowers_to_sample) :    
      sampled_flowers = np.random.poisson(lam = lambda_poisson, size = number_of_patches)

  return(sampled_flowers+1)


def generate_array_procedurally(number_of_flowers,number_of_patches,patchiness_index,environment_size,flowers_per_patch,silent_sim) : 
  """
  Description: 
    Generate an environment of flowers procedurally.
    This function is an algorithm that generates environments with food sites distributed in it.
    The basis of this algorithm is to generate an environment that can be challenging for a species, given its perception range.
    Most of the distances between the flowers and patches are depending on the perception_range and the environment_size.
    Choose them wisely to create a relevant environment for your species.
  Inputs:
    number_of_flowers: total number of flowers 
    patchiness_index: float between 0 (nearly uniform distribution of flowers) and 1 (very patchy), depicting the degree of patchiness_index.
  Outputs:
    array_geometry: pandas dataframe with 4 columns: (flower) ID, x, y, patch (ID)
  """

  perception_range = 10 # Use to be a parameter, but no longer used. As it is needed for this function, it is set as a constant here. Any positive value is fine.

  if patchiness_index >1 or patchiness_index <0 : 
    raise ValueError("Unsupported patchiness index. Values can range between 0 and 1.")
  if number_of_patches > number_of_flowers : 
    raise ValueError("You must have at least one flower per patch.")
  if number_of_flowers == 0 or number_of_patches == 0 :
    raise ValueError("You must at least have one flower and one patch.")
  if (4*perception_range >= environment_size) :
    raise ValueError("The environment size must be at least superior to 4 times the perception range (10 by default).")
  if flowers_per_patch is not None and len(flowers_per_patch)!=number_of_patches : 
    raise ValueError("Vector length of flowers_per_patch is different than the number_of_patches.")

  # Interpatch parameters
  dist_min_to_nest = 2*perception_range
  dist_max_to_nest = environment_size
  dist_nest_increment = dist_max_to_nest/20
  dist_min_patch_patch = 16*perception_range

  # Intrapatch parameters
  dist_min_to_patch = 2*perception_range  # High Patchiness = low min distance to patch center (= Closer flowers in the patch)
  dist_max_to_patch = environment_size - (environment_size-4*perception_range)*patchiness_index  # High Patchiness = low max distance to patch center (= Closer flowers in the patch)
  dist_patch_increment = dist_max_to_patch/20
  dist_min_flower_flower = dist_min_to_patch

  # Initialize output matrices 
  patch_centers = np.zeros((number_of_patches,2))
  array_geometry = np.zeros((number_of_flowers+1,4)) # future columns: ID, x, y, patch

  
  # We create a first patch center
  patch_centers[0,:] = generate_cartesian_coordinates_for_flower(dist_min_to_nest,dist_max_to_nest)
  number_of_patches_generated = 1
  # We create the other patches

  number_of_failed_positioning_in_a_row = 0
  max_failed_positioning_tolerated = 200

  while (number_of_patches_generated < number_of_patches):


    index_of_current_patch = number_of_patches_generated # be careful: indices start from 0
    
    # Creating a patch center
    current_patch_center = generate_cartesian_coordinates_for_flower(dist_min_to_nest,dist_max_to_nest)
    patch_centers[index_of_current_patch,:] = current_patch_center

    # Check if this new patch is far enough from the others
    patch_is_sufficiently_far = check_if_flower_is_sufficiently_far(current_patch_center, dist_min_patch_patch, patch_centers[:(number_of_patches_generated),:])

    if patch_is_sufficiently_far : 
      number_of_patches_generated += 1 # include the new patch
      number_of_failed_positioning_in_a_row = 0

    else : # we failed to position this flower
      number_of_failed_positioning_in_a_row += 1

    if number_of_failed_positioning_in_a_row > max_failed_positioning_tolerated : 
      if not silent_sim : 
        print('generate_array_procedurally failed to position a patch center more than '+str(max_failed_positioning_tolerated)+' times in a row: increasing the range of possible positions by '+str(dist_nest_increment))
      dist_max_to_nest += dist_nest_increment
      number_of_failed_positioning_in_a_row = 0

  # We now have patch centers that follow our rules (distances between patches and to the nest)
  # Now, we want to "populate" these patches. First, we divide the resources between them
  
  # If the flowers are not yet divided into patches:
  if (flowers_per_patch is None) : 
    flowers_per_patch = generate_number_of_flowers_per_patch(number_of_flowers,number_of_patches)
   
  # We now know how many flowers to put at each patch
  # Start filling the array_geometry matrix 

  flower_index = 1 # flower_index 0 corresponds to the nest so we start filling at 1  

  for patch in range (number_of_patches) : 

    patch_center = patch_centers[patch,:] 
    number_of_flowers_in_patch = flowers_per_patch[patch] 
 
    for index_flower_in_patch in range (int(number_of_flowers_in_patch)) : 

      # The first flower in every patch is located at the patch center:
      if index_flower_in_patch == 0 : 
        array_geometry[flower_index,1:3] = patch_center # coordinates of the flower

      # This is another flower whose position must be generated: 
      else : 
        current_flower_coordinates = patch_center + generate_cartesian_coordinates_for_flower(dist_min_to_patch,dist_max_to_patch)
        other_flowers_coordinates = array_geometry[1:(flower_index+1), 1:3] # does not take the nest into account
        
        flower_is_sufficiently_far = check_if_flower_is_sufficiently_far(current_flower_coordinates, dist_min_flower_flower, other_flowers_coordinates)

        number_of_failed_positioning_in_a_row = 0
        max_failed_positioning_tolerated = 200

        while not flower_is_sufficiently_far : 

          number_of_failed_positioning_in_a_row += 1
          current_flower_coordinates = patch_center + generate_cartesian_coordinates_for_flower(dist_min_to_patch,dist_max_to_patch)
          flower_is_sufficiently_far = check_if_flower_is_sufficiently_far(current_flower_coordinates, dist_min_flower_flower, other_flowers_coordinates)

          if number_of_failed_positioning_in_a_row > max_failed_positioning_tolerated : 
            if not silent_sim : 
              print('generate_array_procedurally failed to position a flower more than '+str(max_failed_positioning_tolerated)+' times in a row: increasing the range of possible positions by '+str(dist_patch_increment))
            dist_max_to_patch += dist_patch_increment
            number_of_failed_positioning_in_a_row = 0

        # We have found a correct position for this flower
        array_geometry[flower_index,1:3] = current_flower_coordinates


      array_geometry[flower_index,3] = patch + 1 # index of patch
      flower_index+=1 # position next flower

  array_geometry = pd.DataFrame(array_geometry,columns = ["ID","x","y","patch"])
  return(array_geometry)


def create_environment (array_info, array_number, reuse_generated_arrays,current_working_directory, silent_sim) : 

  '''
  Description:
      Place the nest and flowers in a spatial plane according to the inputs given.
  Inputs: 
      array_info: dictionary with the different characteristics of an array
      array_number: integer, gives the array number
  Outputs: (array_geometry,array_info,array_folder)
  '''

  # This code is used if you generate the environment procedurally
  if (array_info['environment_type'] == "generate") : 
    # Create a name for the folder of the array
    array_info['array_ID'] = create_array_ID(array_info,array_number) 
    # Retrieve the list of arrays available in the Arrays folder
    list_of_known_arrays = np.array(os.listdir(current_working_directory + '\\Arrays'))
    array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']
    # If reuse_generated_arrays, look for similar arrays
    found_similar_array = False 
    if (reuse_generated_arrays) : 
      indices_similar_array = np.where(other_functions.character_match(list_of_known_arrays,array_info['array_ID']))[0]
      found_similar_array = (len(indices_similar_array)!=0)
      if found_similar_array : 
        array_geometry = pd.read_csv(array_folder+'\\array_geometry.csv')
        if not silent_sim : 
          print("A similar array has been found: " + str(list_of_known_arrays[indices_similar_array[0]]) +". Importing.")

    # If not reuse_generated_arrays or there was no similar arrays, generate a new one
    # Adjust arrayID to make sure there is no overlap with an existing array
    if not reuse_generated_arrays or not found_similar_array : 
      index_last_char_array_type_ID = re.search('\_0*\d*',array_info['array_ID']).span()[0]
      array_type_ID = array_info['array_ID'][:index_last_char_array_type_ID]
      array_file_number = get_ID_number_for_array (list_of_known_arrays,array_type_ID)
      array_info['array_ID'] = create_array_ID(array_info,array_file_number)
      array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']

      if not silent_sim :
        print("Generating new array. Array ID is: "+ array_info['array_ID']+".\n")

      os.mkdir(array_folder)
      array_geometry = generate_array_procedurally(array_info['number_of_flowers'],array_info['number_of_patches'],array_info['patchiness_index'],array_info['environment_size'],array_info['flowers_per_patch'],silent_sim) 

      # Write the array and parameters
      array_info_saved = copy.deepcopy(array_info)
      if array_info_saved['flowers_per_patch'] is not None : 
        array_info_saved['flowers_per_patch'] =''.join(map(str,array_info_saved['flowers_per_patch']))
      for key in array_info_saved : 
        array_info_saved[key] = [array_info_saved[key]]
        
      pd.DataFrame(array_info_saved).to_csv(path_or_buf = array_folder + '\\array_info.csv', index = False)
      array_geometry.to_csv(path_or_buf = array_folder + "\\array_geometry.csv", index = False)

  # This code is used if you want to use a specific array
  else :
    if not silent_sim :
      print("Loading known array : " + array_info['environment_type']+"\n")
    array_folder = current_working_directory+"\\Arrays\\"+array_info["environment_type"]
    array_geometry = load_and_normalize_array_geometry(array_folder+"\\array_geometry.csv")
    array_info = pd.read_csv(array_folder+"\\array_info.csv").to_dict(orient='list')
    for key in array_info : 
      array_info[key] = array_info[key][0]
    array_info['array_ID'] = array_info['environment_type']
  return(array_geometry, array_info, array_folder)