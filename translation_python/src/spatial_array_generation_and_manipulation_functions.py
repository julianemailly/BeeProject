'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import os
import numpy as np
import parameters
import geometry_functions
import pandas as pd
import re
#from scipy.stats import poisson

current_working_directory = os.getcwd()

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
  flower_per_patch_string = ''.join(array_info['flower_per_patch'])
  if array_info['number_of_patches'] == 1 : 
    flower_per_patch_string = None
  array_ID = "Array-" + str(array_info["number_of_resources"]) + str(array_info["number_of_patches"]) + "-" + str(array_info["patchiness_index"]) + "-" + flower_per_patch_string + "-" + str(array_info["env_size"]) + "_" + " :02d ".format(array_number)
  return(array_ID)

def get_ID_number_for_array(list_of_known_arrays,array_type_ID):
  """
  Description:
    Scan among known arrays to find a array ID not taken to give to the new array
  Inputs:
    list_of_known_arrays: 
    array_type_ID: 
  Outputs:
    array_ID_number: available ID number
  """
  list_of_similar_arrays = which(character_match(list_of_known_arrays,array_type_ID))
  number_of_similar_arrays = len(list_of_similar_arrays)
  available_numbers = [1 for i in range (number_of_similar_arrays+1)] # available_number[number] = 1 if number is abailable and 0 otherwise
  if number_of_similar_arrays > 0 :
    for i in range (number_of_similar_arrays) : 
      name_of_scanned_array = list_of_similar_arrays[i] 
      end_of_scanned_array = re.findall('\_0*\d*',name_of_scanned_array)[0]
      scanned_array_number = int(re.findall('[^\_0]+',end_of_scanned_array)[0])
      available_numbers[scanned_array_number] = 0 # this number is not available
  array_ID_number = np.where(np.array(available_numbers)==1)[0][0]  # smallest number such as available_numbers[number] == 1 (and add 1 because we want the ID to start from 1)
  return(array_ID_number)

def get_patchy_array(number_of_resources,number_of_patches,patchiness,env_size,flowerPerPatch=None,silent=False) : 
  """
  Description: 
    Generate an environment of flowers procedurally.
    This function is an algorithm that generates environments with food sites distributed in it.
    The basis of this algorithm is to generate an environment that can be challenging for a species, given its perception range.
    Most of the distances between the flowers and patches are depending on the perception_range and the env_size.
    Choose them wisely to create a relevant environment for your species.
  Inputs:
    number_of_resources: 
    patchiness: float between 0 (nearly uniform distribution of flowers) and 1 (very patchy), depicting the degree of patchiness.
  Outputs:
    array_geometry: pandas dataframe with 4 columns: (flower) ID, x, y, patch (ID)
  """

  perception_range = 10 # Used to be a parameter, but no longer used. As it is needed for this function, it is set as a constant here. Any positive value is fine.

  if patchiness >1 or patchiness <0 : 
    raise ValueError("Unsupported Patchiness Index. Values can range between 0 and 1.")
  if number_of_patches > number_of_resources : 
    raise ValueError("You must have at least one resources per patch.")
  if number_of_resources == 0 or number_of_patches == 0 :
    raise ValueError("You must at least have one resource and one patch.")
  if (4*perception_range >= env_size) :
    raise ValueError("The environment size must be at least superior to 4 times the perception range (10 by default).")

  # Interpatch parameters
  dist_min_to_nest = 2*perception_range
  dist_max_to_nest = env_size
  dist_nest_increment = env_size/20
  dist_min_patch_patch = 16*perception_range

  # Initialize output matrices 
  patch_centers = np.zeros((number_of_patches,2))
  array_geometry = pd.DataFrame() # future columns: ID, x, y, patch

  
  # We create a first patch center
  patch_1_dist = np.random.uniform(low=dist_min_to_nest,high=dist_max_to_nest)
  patch_1_azimuth = np.random.uniform(low=0,high=2*np.pi)
  patch_centers[0,:] = geometry_functions.convert_polar_to_cartesian((patch_1_dist,patch_1_azimuth))
  number_of_patches_generated = 1
  
  while (number_of_patches_generated < number_of_patches)
    index_of_current_patch = number_of_patches_generated # be careful: indices start from 0
    
    # Creating a patch center
    distance_nest_patches = np.random.uniform(low=dist_min_to_nest,high=dist_max_to_nest)
    azimuth_nest_patches = np.random.uniform(low=0,high=2*np.pi)
    patch_centers[index_of_current_patch,:] = geometry_functions.convert_polar_to_cartesian((distance_nest_patches,azimuth_nest_patches)) 
    
    # Check if this new patch is far enough from the others
    checking_patch_number = 0
    distance_with_current_patch = geometry_functions.distance(patch_centers[checking_patch_number,:],patch_centers[index_of_patch_to_do,:])
    while (distance_with_current_patch>=dist_min_patch_patch) and (checking_patch_number<number_of_patches_generated) : 
      checking_patch_number+=1
      distance_with_current_patch = geometry_functions.distance(patch_centers[checking_patch_number,:],patch_centers[index_of_patch_to_do,:])

    if (checking_patch_number==number_of_patches_generated) : # if all patches are at a sufficient distance
      number_of_patches_generated += 1 # include the new patch

  
  # We now have patch centers that follow our rules (distances between patches and to the nest)
  # Now, we want to "populate" these patches. First, we divide the resources between them
  
  # We retrieve the remaining number of flowers to put in the array. 
  # We set a poisson distribution to distribute them, with a lambda arbitrarily set to the mean number of flowers per patch.
  if(flower_per_patch is None) : 
   
    population = number_of_resources - number_of_patches # I think that we already have 1 flower per patch 
    lambda_pop = population/number_of_patches 
    patch_pop = np.zeros(number_of_patches)

    # We sample the distribution of the flowers among the patches until we get a distribution fitting the pop size.
    while(np.sum(patch_pop) != population) :    
      patch_pop = np.random.poisson(lam = lambda_pop, size = number_of_patches)
      #patchPop = sample(c(1:population),numberOfPatches,replace=T,prob = dpois(1:population,lambdaPop))
      #poisson_distr = poisson.pmf((np.arange(population)+1),mu=lambda_pop)
      #patch_pop = np.random.choice(a=(np.arange(population)+1),size=number_of_patches,replace=True,p=poisson_distr)

  else :   
    if(len(flower_per_patch)!=number_of_patches) :
      raise ValueError("Vector length of flower_per_patch is different than the number_of_patches.")

    patch_pop = np.array(flower_per_patch)-1 
   
  
  
  # We now know how many flowers to put at each patch 
  i = 1 
  for(patch in 1:numberOfPatches)
   
    # Intrapatch parameters
    distMinToPatch = 2*perceptionRange  # High Patchiness = low min distance to patch center (= Closer flowers in the patch)
    distMaxToPatch = envSize - (envSize-4*perceptionRange)*patchiness  # High Patchiness = low max distance to patch center (= Closer flowers in the patch)
    
    patchCenter = patchCenters[patch,] 
    if(patch==1) i = i + 1 
    if(patch>1) i = i + 1 + patchPop[patch-1] 
    arrayGeometry[i,2:3] = patchCenter 
    arrayGeometry[i,4] = patch+1 
    
    if(patchPop[patch] > 0)
     
      flowerOk = 0 
      failedPositioning = 0 
      while(flowerOk < patchPop[patch])
       
        
        distancePatchFlower = runif(1,distMinToPatch,distMaxToPatch) 
        azimuthPatchFlower = runif(1,0,2*pi) 
        polarVector = c(distancePatchFlower,azimuthPatchFlower) 
        
        cartesianVector = PolarToCartesian(polarVector) 
        arrayGeometry[i+flowerOk+1,2:3] = patchCenter + cartesianVector 
        arrayGeometry[i+flowerOk+1,4] = patch+1 
        
        distToFlowers = c()
        for(flower in 1:(i+flowerOk))
         
          distToFlowers[flower] = DistanceBetween(as.numeric(arrayGeometry[i+1+flowerOk,2:3]),as.numeric(arrayGeometry[flower,2:3]))
         
        
        if(any(distToFlowers < distMinToPatch)) 
         
          failedPositioning = failedPositioning + 1 
          if(failedPositioning > 200) 
           
            cat("GetPatchyArray failed to position a flower in the last 200 iterations. Increasing range of possible positions by ",distNestIncrement,"m.\n",sep="") 
            distMaxToPatch = distMaxToPatch + distNestIncrement 
            failedPositioning = 0 
           
          next
          else  
          flowerOk = flowerOk + 1 
         
       
     
   
  return(arrayGeometry)
 



def create_environment (array_info, array_number, reuse_generated_arrays = parameters.reuse_generated_arrays, silent_sim = parameters.silent_sim) : 

  '''
  Description:
      Place the nest and flowers in a spatial plane according to the inputs given.
  Inputs: 
      array_info: dictionary with the different characteristics of an array
      array_number: integer, gives the array number
  Outputs: (array_geometry,array_info,array_folder)
  '''

  # This code is used if you generate the environment procedurally

  if (array_info['environnement_type'] == "generate") :    

    # Create a name for the folder of the array
    array_info['array_ID'] = create_array_ID(array_info,array_number) 

    # Retrieve the list of arrays available in the Arrays folder
    list_of_known_arrays = os.listdir(current_working_directory + '\\Arrays')
    array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']

    # If reuse_generated_arrays, look for similar arrays
    if (reuse_generated_arrays) : 
      similar_array = which(character_match(list_of_known_arrays,array_info['arrayID']))

      if (len(similar_array)!=0) : 
        array_geometry = pd.read_csv(array_folder)

        if (!silent_sim) : 
          print("A similar array has been found: " + str(list_of_known_arrays[similarArray]) +". Importing.\n")

    # If not reuse_generated_arrays or there was no similar arrays, generate a new one
    # Adjust arrayID to make sure there is no overlap with an existing array
    else : 
      index_last_char_array_type_ID = re.search('\_0*\d*',array_info['array_ID']).span()[0]
      array_type_ID = array_info['array_ID'][:index_last_char_array_type_ID]
      array_file_number = get_ID_number_for_array (list_of_known_arrays,array_type_ID)
      array_info['array_ID'] = create_array_ID(array_info,array_file_number)
      array_folder = current_working_directory + '\\Arrays\\' + array_info['array_ID']

    if (!silent_sim) :
      print("Generating new array. Array ID is: "+ array_info['array_ID']+".\n")

    os.mkdir(array_folder)
    array_geometry = get_patchy_array(array_info['number_of_resources'],array_info['number_of_patches'],array_info['patchiness_index'],array_info['env_size'],array_info['flower_per_patch']) #HERE use get_patchy_array(array_info) directly?

    # Write the array and parameters
    array_info_saved = array_info
    array_info_saved['flower_per_patch'] = ''.join(array_info_saved['flower_per_patch'])
    pd.DataFrame(array_info_saved).to_csv(path = array_folder + '\\array_info.csv', index = F)
    array_geometry.to_csv(path = array_folder + "\\array_geometry.csv", index = F)

  # This code is used if you want to use a specific array
  else :
    if(!silentSim) :
      print("Loading known array : " + array_info['environment_type']+"\n")
    array_folder = current_working_directory+"\\Arrays\\"+array_info["environment_type"]
    array_geometry = load_and_normalize_array_geometry(array_folder+"\\array_geometry.csv")
    array_info = pd.read_csv(array_folder+"\\array_info.csv").to_dict()
    array_info['array_ID'] = array_info['environment_type']

  return(array_geometry, array_info, array_folder)