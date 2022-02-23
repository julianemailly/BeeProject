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
	nrow,ncol = np.shape(matrix)
	normalized_matrix = np.empty((nrow,ncol))
	for row in range (nrow) : 
		row_sum  = np.sum(matrix[row,:])
		if row_sum !=0 : 
			normalized_matrix[row,:]=matrix[row,:]/row_sum
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
	matrix_of_coordinates = array_geometry[["x","y"]] # keep the coordinates
	matrix_of_pairwise_distances = euclidean_distances (matrix_of_coordinates,matrix_of_coordinates)
	return(matrix_of_pairwise_distances)



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


def compute_route_quality(number_of_flowers_foraged,route_length) : 
	"""
	Description:
		Route quality evaluation function
	Inputs:
		number_of_flowers_foraged: number of flowers collected
		route_length: distance covered by the bee
	Outputs:
		Route quality (float)
	"""
	if route_length == 0 :
		return(0)
	else :
		return(number_of_flowers_foraged**2/route_length)