
import numpy as np

class Particle:
	"""A particle class for a FastSLAM 2.0 implementation with unknown data correspondence. 
		Map can have unlimited features
		
		Member variables:
		state (np.array)	--The pose of the particle (x,y,theta)
		N     (int)			--The number of landmarks in the map
		feats (np.array)	--The means of the landmarks (4xN)
		covs  (np.array)	--The covariances of the landmarks (2x2xN)
		weight(float)		--The importance weight of the particle
	"""

	def __init__(self,x,y,theta,wt):
		"""Instantiate a particle whose state will be pose"""
		self.state = np.array([x,y,theta],dtype=np.double)
		# No landmarks observed
		self.N=0
		# featId, x,y location in map, count
		self.feats = np.array((4,1), dtype=np.double)
		# Array of 2x2 covariances
		self.covs = np.empty((2,2,1), dtype=np.double)
		# Particle weight
		self.weight = wt

	def updateState(self, x,y,theta):
		"""Update the state of the particle"""
		self.state[0]=x
		self.state[1]=y
		self.state[2]=theta

	def observed(self,featID):
		"""Check if the feature has been observed already
			Parameters:
			featID (int)	--ID of the observed feature
			
			Returns:
			observed (bool)
		"""
		for i in range(self.N):
			if(self.feats[0,i]==featID):
				return True
		return False

	def initFeat(self, featID, mu, cov, w):
		"""Initialize a feature's mean, covariance and weight
			Parameters:
			featID (int)	--ID of the observed feature
			mu (np.array)	--location of landmark (x,y)
			cov (np.array)	--covariance matrix
			w (float)		--importance weight
		"""
		print("Init Feat")
		#np arrays append creates a new array, last arg is what axis to insert along
		print("Feats: ", self.feats)
		# initalize counter
		self.feats = np.append(self.feats,[featID,mu[0],mu[1],1],0)
		print("New Feats: ", self.feats)
		self.covs= np.append(self.covs, cov,0)
		self.N+=1
		self.weight=w

	def updateFeat(self, featID, mu, cov, w):
		"""Update a feature's mean, covariance counter and weight
			Parameters:
			featID (int)	--ID of the observed feature
			mu (np.array)	--location of landmark (x,y)
			cov (np.array)	--covariance matrix
			w (float)		--importance weight of the particle
		"""
		for i in range(N):
			if(self.feats[0,i]==featID):
				self.feats[1,i]=mu[0]
				self.feats[2,i]=mu[1]
				# increment counter
				self.feats[3,i]+=1
				self.covs[,,i]=cov
				self.weight=w
				break
		

	def getFeat(self, featID):
		"""Get a feature's mean
			Parameters:
			featID (int)	--ID of the observed feature
			Returns:
			mu (np.array)	--location of landmark (x,y)
		"""
		for i in range(self.N):
        	if (self.feats[0,i] ==featID):
        		return self.feats[1:3,i]

	def decrementFeat(self, featID):
		"""Decrement feature counter and check if valid
			Parameters:
			featID (int)	--ID of the feature
		"""
		for i in range(self.N):
        	if (self.feats[0,i] ==featID):
        		# decrement counter
        		self.feats[3,i]-=1
        		# discard dubious features
        		if (self.feats[3,i] < 0):
        			self.feats = np.delete(self.feats, i)
        			self.covs = np.delete(self.covs, i)
        			self.N-=1
