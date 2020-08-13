
import numpy as np

class Particle:
    """A particle class for a FastSLAM 2.0 implementation with unknown data correspondence. 
        Map can have unlimited features
        
        Member variables:
        state (np.array)    --The pose of the particle (x,y,theta)
        N     (int)         --The number of landmarks in the map
        feats (np.array)    --The means of the landmarks (4xN)
        covs  (np.array)    --The covariances of the landmarks (2x2xN)
        weight(float)       --The importance weight of the particle
    """

    def __init__(self,x,y,theta,wt):
        """Instantiate a particle whose state will be pose"""
        self.state = np.array([x,y,theta],dtype=np.double)
        # No landmarks observed
        self.N=0
        # x,y location in map, count, featID is just index
        self.feats = np.empty((3,1), dtype=np.double)
        # Array of 2x2 covariances
        self.covs = np.empty((2,2,1), dtype=np.double)
        # Particle weight
        self.weight = wt

    def updateState(self, x,y,theta):
        """Update the state of the particle"""
        self.state[0]=x
        self.state[1]=y
        self.state[2]=theta

    def initFeat(self, featID, mu, cov, w):
        """Initialize a feature's mean, covariance and counter as well as update weight
            Parameters:
            featID (int)    --ID of the observed feature
            mu (np.array)   --location of landmark (x,y)
            cov (np.array)  --covariance matrix
            w (float)       --importance weight
        """
        # note, no need to update N, it will have been update earlier
        
        # first feature
        if (self.N == 1):
            #print("init first feat: ")
            self.feats = np.array([[mu[0]],
                                   [mu[1]],
                                   [1]],dtype = np.double)
            self.covs=cov.reshape(2,2,1)
        
        # additional features
        else:
            #print("init next feat: ")
            #np arrays append creates a new array, last arg is what axis to insert along
            self.feats = np.append(self.feats,np.array([[mu[0]],
                                                        [mu[1]],
                                                        [1]]),1)
            self.covs= np.append(self.covs,cov.reshape(2,2,1),2)
    
        self.weight=w

    def updateFeat(self, featID, mu, cov, w):
        """Update a feature's mean, covariance counter and weight
            Parameters:
            featID (int)    --ID of the observed feature
            mu (np.array)   --location of landmark (x,y)
            cov (np.array)  --covariance matrix
            w (float)       --importance weight of the particle
        """
        # change to indexing from 0
        featID-=1
        self.feats[0,featID]=mu[0]
        self.feats[1,featID]=mu[1]
        # increment counter
        self.feats[2,featID]+=1
        self.covs[:,:,featID]=cov
        self.weight=w
        
    def getFeat(self, featID):
        """Get a feature's mean
            Parameters:
            featID (int)    --ID of the observed feature
            Returns:
            mu (np.array)   --location of landmark (x,y)
        """
        #print("featID: ", featID)
        #print("Feats: ", self.feats)

        featID-=1
        return self.feats[0:2,featID]

    def getCov(self, featID):
        """Get a feature's covariance
            Parameters:
            featID (int)    --ID of the observed feature
            Returns:
            cov (np.array)  -- covariance of landmark location (x,y)
        """
        featID-=1
        return self.covs[:,:,featID]

    def decrementFeat(self, featID):
        """Decrement feature counter and check if valid
            Parameters:
            featID (int)    --ID of the feature
        """
        featID-=1
        # decrement counter
        self.feats[2,featID]-=1
        # discard dubious features
        if (self.feats[2,featID] < 0):
            self.N-=1
            self.feats = np.delete(self.feats, featID,1)
            self.covs = np.delete(self.covs, featID,2)
                    
                
