import numpy as np

class Particle:
	"""A particle class for a fastSLAM implementation. Right now map only has one feature
		Member variables
		state (np.array)	--The pose of the particle (x,y,theta)
		feats (np.array)	--The means of the landmarks
		cov	  (np.array)	--The covariances of the landmarks
	"""
	def __init__(self):
		"""Instantiate a particle whose state will always be pose"""
		self.state = np.empty([3,1], dtype=np.double)
		# ID and Location of feature in cartesian coordinate system
		self.feats = np.empty([3,1], dtype=np.double)
		# Array of 2x2 covariances
		self.covs = np.empty([2,2], dtype=np.double)
		self.weight = 0

	def __init__(self,x,y,theta,wt):
		"""Instantiate a particle whose state will be pose"""
		self.state = np.array([x,y,theta],dtype=np.double)
		# Location of feature in cartesian coordinate system
		self.feats = np.array([0,0,0], dtype=np.double)
		# Array of 2x2 covariances
		self.covs = np.empty([2,2,1], dtype=np.double)
		# Particle weight
		self.weight = wt

	def updateState(self, x,y,theta):
		"""Update the state of the particle"""
		self.state[0]=x
		self.state[1]=y
		self.state[2]=theta

	def observed(self,featID):
		"""Check if the feature has been observed
			Parameters:
			featID (int)	--ID of the observed feature
			Returns:
			observed (bool)
		"""
		
		if(self.feat[0]==featID):
			return true

		return false

	def initFeat(self, featID, mu, cov, w):
		"""Initialize a feature's mean, covariance and weight
			Parameters:
			featID (int)	--ID of the observed feature
			mu (np.array)	--location of landmark (x,y)
			cov (np.array)	--covariance matrix
			w (float)		--importance weight
		"""

		# Can keep simple, since we know only one feature
		self.feat[0]=featID
		self.feat[1]=mu[0]
		self.feat[2]=mu[1]
		self.cov=cov
		self.weight=w

	def updateFeat(self, featID, mu, cov, w):
		"""Update a feature's mean, covariance and weight
			Parameters:
			featID (int)	--ID of the observed feature
			mu (np.array)	--location of landmark (x,y)
			cov (np.array)	--covariance matrix
		"""

		# Can keep simple, since we know only one feature
		self.feat[1]=mu[0]
		self.feat[2]=mu[1]
		self.cov=cov
		self.weight=w