'''
Author: Juliane Mailly
Contact: julianemailly0gmail.com
'''

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Geometry Functions --------------------------------------------------------------


def convert_polar_to_cartesian (polar_vector) : 
	'''
	Description:
		Converts a polar vector (rho, theta) in a cartesian vector
	Inputs:
		polar_vector: can be a tuple or a list
	Outputs:
		cartesian_vector: a list
	'''
	cartesian_vector = [polar_vector[0] * np.cos(polar_vector[1]) , polar_vector[0] * np.sin(polar_vector[1])]
	return(cartesian_vector)


def normalize_matrix_by_row (matrix) : 
	"""
	Description:
		Normalizes a matrix such that the sum of the value on each row equals 1
	Inputs: 
		matrix: a float matrix 
	Outputs: 
		normalized_matrix: the matrix normalized
	"""
	row_sums = matrix.sum(axis=1)
	normalized_matrix = matrix / row_sums[:, np.newaxis]
	return(normalized_matrix)


def distance(point1,point2) : 
	"""
	Description:
		Gives the euclidean distance between to points in a 2D space.
	Inputs: 
		point1, point2: two points
	Outputs: 
		Euclidean distance between point1 and point2
	"""
	return(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))


def get_matrix_of_distances_between_flowers(array_geometry) : 
	"""
	Description:
		Given the position of flowers, compute the mtrix of pairwise distances between flowers
	Inputs: 
		array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
	Outputs: 
		matrix_of_pairwise_distances: matrix of size number_of_flowers*number_of_flowers gibing the euclidean distance between pairs of flowers
	"""
	matrix_of_coordinates = array_geometry.iloc[:,1:3] # keep the coordinates
	matrix_of_pairwise_distances = euclidean_distances (matrix_of_coordinates,matrix_of_coordinates)
	return(matrix_of_pairwise_distances)


def get_route_length(route,array_geometry) : 
	"""
	Inputs: 
		route: vector of the index of visited flowers during a bout
		array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
	Outputs: 
		route_length: total length of the route
	"""
	matrix_of_pairwise_distances = get_matrix_of_distances_between_flowers(array_geometry)
	number_of_flowers_in_route = len(route)
	route_length = 0
	for flower_index in range (number_of_flowers_in_route-1) : 
		previous_flower = route[flower_index]
		next_flower = route[flower_index+1]
		route_length += matrix_of_pairwise_distances[previous_flower,next_flower]
	return(route_length)


def give_probability_of_vector_with_dist_factor(x,dist_factor) :
  """
  Description:
    Gets the probability of x given the function [probability = 1/x^distFactor]
  Inputs:
    x: matrix (usually matrix containing the distances between flowers)
    dist_factor: float, parameter to estimate the probabilities 
  Outputs:
    A vector of probabilities # COMMENT: not probabilities because not normalized!
  """ 
  dimensions = np.shape(x)
  probabilities = np.zeros(dimensions)
  for i in range (dimensions[0]) : 
  	for j in range (dimensions[1]) : 
  		if x[i,j] != 0 : 
  			probabilities[i,j] = 1/x[i,j]**dist_factor
  return (probabilities)


def initialize_probability_matrix_list(array_geometry,dist_factor,number_of_bees) : 
  """
  Description: 
    Generates the probability matrix for a given array for each bee
  Inputs:
    array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
    dist_factor: float, parameter to estimate the probabilities 
    number_of_bees: integer giving the total number_of_bees
  Outputs: 
    List of probability matrices (one for each bee)
  """
  distance_between_flowers = get_matrix_of_distances_between_flowers(array_geometry)
  probability_matrix = normalize_matrix_by_row (give_probability_of_vector_with_dist_factor(distance_between_flowers,dist_factor))
  return ([probability_matrix for bee in range (number_of_bees)])


def compute_route_quality(number_of_resources_foraged,route_length) : 
	"""
	Description:
		Route quality evaluation function
	Inputs:
		number_of_resources_foraged: number of flowers collected
		route_length: distance covered by the bee
	Outputs:
		Route quality (float)
	"""
	return(number_of_resources**2/route_length)


def get_route_quality(route,array_geometry,list_of_resources_foraged) : 
	"""
	Description:
		Computes the route quality of a given visitation sequence
	Inputs:
		route: list of indices of flowers visited during a bout
		array_geometry: pandas dataframe of size 4*number_of_flowers : flower ID, x, y, patch ID
		list_of_resources_foraged: list of how many resources were foraged on each flower of the route
	Outputs: 
		Route quality (float)
	"""
	if len(route)==0 : 
		return(0)
	else : 
		route_length = get_route_length(route,array_geometry)
		number_of_foraged_resources = np.sum(list_of_resources_foraged)
		route_quality = compute_route_quality(number_of_foraged_resources,route_length)
		return(route_quality)


def formating_route(route):
	"""
	Description: 
		Takes a route such as [0,n1,n2,...,np,0,...,0] and remove the unecessary 0 to make [0,n1,n2,...,np,0]
	Inputs:
		route: [0,n1,n2,...,np,0,...,0]
	Outputs:
		route: [0,n1,n2,...,np,0]
	"""
	route=np.array(route)
	route_length = len(route)
	i = route_length - 1
	while i!=0 and route[i] == 0:
		route = np.delete(route,i)
		i=i-1
	route = np.concatenate((route,[0]))
	return(route)