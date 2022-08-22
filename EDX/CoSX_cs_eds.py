from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from utils_cs_eds import *
import hyperspy.api as hs
from tqdm import tqdm 
import numpy as np
import ctvlib
import h5py

################### PARAMETERS #######################

fname = 'CoSX/Interface_maps.h5'; mapNum = 'map7/'

# Convergence Parameters
gamma = 1.6; lambdaEDS = 0.005; nIter = 30; bkg = 1e-1

# TV Min Parameters
regularize = True; ng = 15; lambdaTV = 0.1; 

saveCostFunction = False; saveImages = True

###################################################

# Load Raw Data and Reshape
file = h5py.File(fname, 'r')

# Parse Chemical Maps
elementList = ['Co', 'O', 'S']

xx = np.array([],dtype=np.float32)
for ee in elementList:
	edsMap = file[mapNum+ee][:,:]
	edsMap -= np.min(edsMap); edsMap /= np.max(edsMap)
	xx = np.concatenate([xx,edsMap.flatten()])
xx0 = xx.copy()

# Image Dimensions
(nx, ny) = edsMap.shape; nPix = nx * ny
nz = len(elementList); lambdaHAADF = 1/nz

# C++ TV Min Regularizers
reg = ctvlib.ctvlib(nx, ny)

# HAADF Signal (Measurements)
b = file[mapNum+'HAADF'][:].flatten()

# Data Subtraction and Normalization 
b -= np.min(b); b /= np.max(b)

# Create Measurement Matrix
A = create_measurement_matrix(nx,ny,nz)

# # Show Raw Data
# plot_elemental_images(xx, b, elementList, nx, ny, 2,2)
# save_images(xx, b, elementList, nx, ny)
# exit()

# Auxiliary Functions
lsqFun = lambda inData : 0.5 * np.linalg.norm(A.dot(inData**gamma) - b) **2
poissonFun = lambda inData : np.sum(xx0 * np.log(inData + 1e-8) - inData)

# Main Loop
costHAADF = np.zeros(nIter,dtype=np.float32); costEDS = np.zeros(nIter, dtype=np.float32); costTV = np.zeros(nIter, dtype=np.float32);
for kk in tqdm(range(nIter)):

	# HAADF Update
	xx -=  gamma * spdiags(xx**(gamma - 1), [0], nz*nx*ny, nz*nx*ny) * lambdaHAADF * A.transpose() * (A.dot(xx**gamma) - b) + lambdaEDS * (1 - xx0 / (xx + bkg))
	xx[xx<0] = 0

	# Regularization 
	if regularize:
		for zz in range(nz):
			xx[zz*nPix:(zz+1)*nPix] = reg.fgp_tv( xx[zz*nPix:(zz+1)*nPix], lambdaTV, ng)

	# Measure Cost Function
	costHAADF[kk] = lsqFun(xx); costEDS[kk] = poissonFun(xx)

	# Measure Isotropic TV 
	if regularize:
		for zz in range(nz):
			costTV[kk] += reg.tv( xx[zz*nPix:(zz+1)*nPix] )

# Show Reconstructed Signal
plot_elemental_images(xx,A.dot(xx**gamma),elementList,nx,ny,2,2)

# Display Cost Functions and Descent Parameters
plot_convergence(costHAADF, lambdaHAADF, costEDS, lambdaEDS, costTV, lambdaTV)

# Save Images
if saveImages: save_images(xx, A.dot(xx**gamma), elementList, nx, ny)

# Save Cost Functions (Experiment)
if saveCostFunction:
	import h5py 
	file = h5py.File('results/CoSX_convergence.h5', 'w')
	file.create_dataset('costHAADF', data=costHAADF); file['lambdaHAADF'] = lambdaHAADF
	file.create_dataset('costEDS', data=costEDS); file['lambdaEDS'] = lambdaEDS
	file.create_dataset('costTV', data=costTV); file['lambdaTV'] = lambdaTV
	file['gamma'] = gamma; file.close()