from multimodal_fusion import utils
from scipy.sparse import spdiags
from tqdm import tqdm
import numpy as np
import requests
import io
import h5py

try: 
    from multimodal_fusion import ctvlib
    cRegFlag = True
except: 
    from multimodal_fusion import pyreg as ctvlib
    cRegFlag = False
    print("Warning: C-libraries not found, using Python implementation of regularization.")

class DataFusion:
    def __init__(self, elements, gamma=1.6, regularize=True):
        """
        Initialize the DataFusion object.

        Args:
            elements (list): List of strings elements to be fused.
            gamma (float): Exponent for the gamma-divergence.
            regularize (bool): Whether to use regularization.
        """

        # Default Optimization Parameters
        self.gamma = gamma
        self.regularize = regularize

        # Initialize Variables
        self.xx = None
        self.b = None
        self.elements = elements

        # Get the Periodic Table
        self.pt_table = utils.get_periodic_table()
    
    def load_chemical_maps(self, chemical_maps, method = 0):
        """
        Load chemical maps and create measurement matrix.

        Args:
            chemical_maps (dict): Dictionary of chemical maps.
        """
        
        # Concatenate Chemical Maps
        self.xx = np.array([],dtype=np.float32)
        self.nz = len(chemical_maps)
        for ee in chemical_maps:
            edsMap = chemical_maps[ee]
            edsMap -= np.min(edsMap); 
            edsMap /= np.max(edsMap)
            self.xx = np.concatenate([self.xx,edsMap.flatten()])
        
        # Parse Image Dimensions and Create Measurement Matrix
        (self.nx, self.ny) = edsMap.shape
        self.nPix = self.nx * self.ny

        # Default is the Unweighted Measurement Matrix
        zNums = list(map(lambda x: self.pt_table[x.lower()], self.elements))
        self.A = utils.create_weighted_measurement_matrix(
            self.nx, self.ny, self.nz, zNums, self.gamma, method)

        # Initialize Regularization
        self.reg = ctvlib(self.nx, self.ny)

        # Make Copy of Raw Measurements for Poisson Maximum Likelihood Term 
        self.xx0 = self.xx.copy()

    def load_haadf(self, haadf):
        """
        Load HAADF data and normalize.

        Args:
            haadf (numpy.ndarray): HAADF data.
        """
        self.b = haadf.flatten()
        self.b -= np.min(self.b); 
        self.b /= np.max(self.b)

    def run(self, nIter = 50, lambdaEDS = 0.005, lambdaTV = 0.1, ng = 15, bkg = 0.1, plot_images = True, plot_convergence = True):
        """
        Run the data fusion algorithm.

        Args:
            nIter (int): Number of iterations.
            lambdaHAADF (float): Lambda for HAADF data.
            lambdaEDS (float): Lambda for EDS data.
            ng (int): Number of gradient descent iterations.
            lambdaTV (float): Lambda for TV regularization.
            bkg (float): Background level.
            plot_images (bool): Whether to plot the images.
            plot_convergence (bool): Whether to plot the convergence.
        """

        # Default lambdaHAADF is inverse of number of elements
        lambdaHAADF = 1/self.nz

        # Check if Data is Loaded
        if self.b is None:
            raise ValueError("HAADF data not loaded. Please call load_haadf() first.")
        if self.xx is None:
            raise ValueError("Chemical maps not loaded. Please call load_chemical_maps() first.")

        # Auxiliary Functions
        lsqFun = lambda inData : 0.5 * np.linalg.norm(self.A.dot(inData**self.gamma) - self.b) **2
        poissonFun = lambda inData : np.sum(self.xx0 * np.log(inData + 1e-8) - inData)

        # Initialize Cost Functions
        if plot_convergence:
            self.costHAADF = np.zeros(nIter,dtype=np.float32)
            self.costEDS = np.zeros(nIter, dtype=np.float32)
            self.costTV = np.zeros(nIter, dtype=np.float32)

        # Main Loop
        self.xx = self.xx0.copy() # Reset the Inital Guess
        for kk in tqdm(range(nIter)):

            # HAADF Update
            self.xx -=  self.gamma * spdiags(self.xx**(self.gamma - 1), [0], self.nz*self.nx*self.ny, self.nz*self.nx*self.ny) * lambdaHAADF * self.A.transpose() * (self.A.dot(self.xx**self.gamma) - self.b) + lambdaEDS * (1 - self.xx0 / (self.xx + bkg))
            self.xx[self.xx<0] = 0

            # Regularization 
            if self.regularize:
                for zz in range(self.nz):
                    if cRegFlag: self.xx[zz*self.nPix:(zz+1)*self.nPix] = self.reg.fgp_tv( self.xx[zz*self.nPix:(zz+1)*self.nPix], lambdaTV, ng)	
                    else :  	 self.xx[zz*self.nPix:(zz+1)*self.nPix] = self.reg.fgp_tv( self.xx[zz*self.nPix:(zz+1)*self.nPix].reshape(self.nx,self.ny), lambdaTV, ng).flatten()
                    
            # Measure Cost Function
            if plot_convergence:
                self.costHAADF[kk] = lsqFun(self.xx)
                self.costEDS[kk] = poissonFun(self.xx)

            # Measure Isotropic TV 
            if self.regularize and plot_convergence:
                for zz in range(self.nz):
                    self.costTV[kk] += self.reg.tv( self.xx[zz*self.nPix:(zz+1)*self.nPix].reshape(self.nx,self.ny) )

        # Show Reconstructed Signal
        if plot_images:
            utils.plot_elemental_images(self.xx,self.A.dot(self.xx**self.gamma),self.elements,self.nx,self.ny)

        # Display Cost Functions and Descent Parameters
        if plot_convergence:
            utils.plot_convergence(self.costHAADF, lambdaHAADF, self.costEDS, lambdaEDS, self.costTV, lambdaTV)

    def get_results(self):
        """
        Get the results of the data fusion algorithm.

        Returns:
            dict: Dictionary of results.
        """
        results = {}
        for ii in range(self.nz):
            results[self.elements[ii]] = self.xx[ii*self.nPix:(ii+1)*self.nPix].reshape(self.nx,self.ny)
        return results
    
    def save_results(self, path):
        """
        Save the results of the data fusion algorithm.

        Args:
            path (str): Path to save the results.
        """
        utils.save_images(self.xx, self.b, self.elements, self.nx, self.ny)

    def load_edx_example(self, map_index = 7):
        """
        Load starter data from github repository.

        Args: 
            map_index (int, optional): Which of the 6 available datasets to load (zero indexed). Defaults to zero.
        """

        # Load url for dataset
        url = "https://raw.githubusercontent.com/jtschwar/Multi-Modal-2D-Data-Fusion/main/multimodal_fusion/example_data/demo_EDX_maps.h5"       
        response = requests.get(url)

        # Prepare path within h5py data 
        map_path = f"/map{map_index}"

        # If url is found, read in the data without saving to disk, and load Co, S, O, and HAADF.
        if response.status_code == 200:
            h5_data = h5py.File(io.BytesIO(response.content), "r")
            cobalt_map = h5_data[map_path]['Co'][:]
            sulfur_map = h5_data[map_path]['S'][:]
            oxygen_map = h5_data[map_path]['O'][:]
            haadf_map = h5_data[map_path]['HAADF'][:]
            h5_data.close()
            return cobalt_map, sulfur_map, oxygen_map, haadf_map
        # If url is not found, exit function and return zero.
        else:
            print("Failed to download:", response.status_code)
            return 0
