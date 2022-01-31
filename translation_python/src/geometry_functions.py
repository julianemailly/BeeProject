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
	Inputs:
		polar_vector: can be a tuple or a list
	Outputs:
		cartesian_vector: a list
	'''
	cartesian_vector = [polar_vector[0] * np.cos(polar_vector[1]) , polar_vector[0] * np.sin(polar_vector[1])]
	return(cartesian_vector)


def normalize_matrix_by_row (matrix) : 
	row_sums = matrix.sum(axis=1)
	normalized_matrix = matrix / row_sums[:, np.newaxis]
	return(normalized_matrix)


def distance(point1,point2) : 
	return(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))


def get_matrix_of_distances_between_flowers(array_geometry) : 
	"""
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