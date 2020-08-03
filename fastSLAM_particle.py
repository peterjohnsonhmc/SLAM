import numpy as np

class Particle:
	"""A particle class for a fastSLAM implementation
		Member variables
		state (np.array)	--The pose of the particle (x,y,theta)
		map	  (np.array)	--The means of the landmarks
		cov	  (np.array)	--The covariances of the landmarks
	"""
	def __init__(self):
		"""Instantiate a particle whose state will always be pose"""
		self.state = np.empty(3,1)
		# Location of feature in cartesian coordinate system
		self.feats = np.empty(2,1)
		# Array of 2x2 covariances
		self.covs = np.empty(2,2,1)
		self.weight = 0
