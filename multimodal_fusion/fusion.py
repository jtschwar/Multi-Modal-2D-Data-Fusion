from multimodal_fusion import utils
from scipy.sparse import spdiags
from tqdm import tqdm
import numpy as np

try: 
    from multimodal_fusion import ctvlib
    cRegFlag = True
except: 
    cRegFlag = False
    print("Warning: C-libraries not found, using Python implementation of regularization.")

class DataFusion:
    def __init__(self, image, num_elements):
        self.nx, self.ny = image.shape
        self.nz = num_elements
        self.A = utils.create_measurement_matrix(self.nx, self.ny, self.nz)

        # Default Optimization Parameters
        self.gamma = 1.6
        self.regularize = True

    def load_chemical_maps(self, elements):
        self.xx = np.array([],dtype=np.float32)
        for ee in elements:
            edsMap = self.elements[ee]
            edsMap -= np.min(edsMap); edsMap /= np.max(edsMap)
            self.xx = np.concatenate([self.xx,edsMap.flatten()])
        
        # Make Copy of Raw Measurements for Poisson Maximum Likelihood Term 
        self.xx0 = self.xx.copy()

    def run(self, nIter = 50, lambdaHAADF = 0.005, lambdaEDS = 0.005, ng = 15, lambdaTV = 0.1):

        # Auxiliary Functions
        lsqFun = lambda inData : 0.5 * np.linalg.norm(self.A.dot(inData**self.gamma) - b) **2
        poissonFun = lambda inData : np.sum(self.xx0 * np.log(inData + 1e-8) - inData)

        # Main Loop
        costHAADF = np.zeros(nIter,dtype=np.float32); costEDS = np.zeros(nIter, dtype=np.float32); costTV = np.zeros(nIter, dtype=np.float32);
        for kk in tqdm(range(nIter)):

            # HAADF Update
            xx -=  self.gamma * spdiags(xx**(self.gamma - 1), [0], self.nz*self.nx*self.ny, self.nz*self.nx*self.ny) * lambdaHAADF * self.A.transpose() * (self.A.dot(xx**self.gamma) - b) + lambdaEDS * (1 - xx0 / (xx + bkg))
            xx[xx<0] = 0

            # Regularization 
            if self.regularize:
                for zz in range(self.nz):
                    if cRegFlag: xx[zz*nPix:(zz+1)*nPix] = reg.fgp_tv( xx[zz*nPix:(zz+1)*nPix], lambdaTV, ng)	
                    else :  	 xx[zz*nPix:(zz+1)*nPix] = reg.fgp_tv( xx[zz*nPix:(zz+1)*nPix].reshape(self.nx,self.ny), lambdaTV, ng).flatten()
                    
            # Measure Cost Function
            costHAADF[kk] = lsqFun(xx); costEDS[kk] = poissonFun(xx)

            # Measure Isotropic TV 
            if self.regularize:
                for zz in range(self.nz):
                    costTV[kk] += reg.tv( xx[zz*nPix:(zz+1)*nPix].reshape(self.nx,self.ny) )

        # Show Reconstructed Signal
        utils.plot_elemental_images(xx,self.A.dot(xx**self.gamma),elementList,self.nx,self.ny,2,2)

        # Display Cost Functions and Descent Parameters
        utils.plot_convergence(costHAADF, lambdaHAADF, costEDS, lambdaEDS, costTV, lambdaTV)
