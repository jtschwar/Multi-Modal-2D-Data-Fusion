from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from skimage.io import imsave
from tqdm import tqdm
import numpy as np

def calculate_subplot_dimensions(n_total):
    """
    Calculate optimal subplot dimensions for n_total images.
    
    Args:
        n_total (int): Total number of images to display
        
    Returns:
        tuple: (nrows, ncols) for subplot arrangement
    """
    if n_total <= 2:
        return (1, n_total)
    elif n_total <= 4:
        return (2, 2)
    elif n_total <= 6:
        return (2, 3)
    elif n_total <= 9:
        return (3, 3)
    elif n_total <= 12:
        return (3, 4)
    else:
        # For larger numbers, aim for roughly square arrangement
        ncols = int(np.ceil(np.sqrt(n_total)))
        nrows = int(np.ceil(n_total / ncols))
        return (nrows, ncols)

def plot_elemental_images(data,haadf,eList,nx,ny, nrows=None, ncols=None):

	# Auto-calculate dimensions if not provided
	n_total = len(eList) + 1  # +1 for HAADF
	if nrows is None or ncols is None:
		nrows, ncols = calculate_subplot_dimensions(n_total)

	fig, ax = plt.subplots(nrows,ncols,figsize=(12,8))
	ax = ax.flatten()

	ax[0].imshow(haadf.reshape(nx,ny)); ax[0].set_title('HAADF'); ax[0].axis('off')
	for ii in range(len(eList)):
		ax[ii+1].imshow(data[ii*(nx*ny):(ii+1)*(nx*ny)].reshape(nx,ny),cmap='gray'); ax[ii+1].set_title(eList[ii]); ax[ii+1].axis('off')
	plt.show()

def plot_cost_function(data, save_plot):
	plt.plot(data)
	plt.xlabel('Iteration #')
	plt.ylabel('Cost Function')
	if save_plot:  plt.savefig('results/cost_fun.png')
	plt.show()

def save_images(data, haadf, eList, nx, ny):

	imsave('results/haadf_recon.tif', haadf.reshape(nx,ny))
	for ii in range(len(eList)):
		imsave('results/{}_signal.tif'.format(eList[ii]), data[ii*(nx*ny):(ii+1)*(nx*ny)].reshape(nx,ny))

def plot_convergence(costLS, L_LS, costP, L_PS, costTV, L_TV):
	fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(12,6))

	nIter = costLS.shape[0]

	ax1.plot(costLS)
	ax1.set_ylabel(r'$\frac{1}{2} ||Ax^{\gamma} - b||^2$')
	ax1.set_title('Step Size: ' + str(L_LS))
	ax1.set_xticklabels([])
	ax1.set_xlim([0,nIter-1])
	ax1.grid(True)

	ax2.plot(costP)
	ax2.set_ylabel(r'$\sum (x - b \cdot \log(x))$')
	ax2.set_title('Step Size: ' + str(L_PS))
	ax2.set_xticklabels([])
	ax2.set_xlim([0,nIter-1])
	ax2.grid(True)

	ax3.plot(costTV)
	ax3.set_ylabel(r'$\sum ||x||_{TV}$')
	ax3.set_xlabel('Iteration #')
	ax3.set_title('Step Size: ' + str(L_TV))
	ax3.set_xlim([0,nIter-1])
	ax3.grid(True)

	plt.show()

def create_measurement_matrix(nx, ny, nz):
	#Create Measurement Matrix.
	vals = np.zeros([nz*ny*nx], dtype=int)
	row =  np.zeros([nz*ny*nx], dtype=int)
	col =  np.zeros([nz*ny*nx], dtype=int)
	vals[:] = 1

	ii = 0; ind = 0
	while ii < nz*nx*ny:
		for jj in range(nz):
			row[ii+jj] = ind
			col[ii+jj] = ind + nx*ny*jj

		ii += nz
		ind += 1
	A = csr_matrix((vals, (row, col)), shape=(nx*ny, nz*nx*ny), dtype=np.float32)

	return A

def create_weighted_measurement_matrix(nx, ny, nz, zNums, gamma,method=0):
	#Create Measurement Matrix.
	vals = np.zeros([nz*ny*nx], dtype=np.float16)
	row =  np.zeros([nz*ny*nx], dtype=int)
	col =  np.zeros([nz*ny*nx], dtype=int)
	vals[:] = 1

	ii = 0; ind = 0
	while ii < nz*nx*ny:
		for jj in range(nz):
			row[ii+jj] = ind
			col[ii+jj] = ind + nx*ny*jj
	
			if method == 0:
				pass 
			if method == 1:
				vals[ii+jj] = zNums[jj] / np.mean(zNums) # Z_{i}^{gamma} / ( sum_{i} ( Z_i ) / Nelements )
			if method == 2:
				vals[ii+jj] = zNums[jj]**gamma / np.mean(zNums**gamma) # Z_{i}^{gamma} / ( sum_{i} ( Z_i^{gamma} ) / Nelements )
			if method == 3:
				vals[ii+jj] = zNums[jj] / np.sum(zNums) # Z_{i} / sum_{i} Z_i
			if method == 4:
				vals[ii+jj] = zNums[jj]**gamma / np.sum(zNums**gamma) # Z_{i}^{gamma} / sum_{i} ( Z_i^{gamma} )

		ii += nz
		ind += 1
	A = csr_matrix((vals, (row, col)), shape=(nx*ny, nz*nx*ny), dtype=np.float32)

	return A

def calculate_curvature(dataX, dataY):
    '''data is assumed to be same size
	Uses Wendy's method'''

    # Shape Data appropriately
    n = np.shape(dataX)[0]
    d = np.column_stack((dataX,dataY))
    K = np.zeros(n-2) # Curvature Vector

    for i in np.arange(n-2):
        x = d[i:i+3,0]
        y = d[i:i+3,1]

        K[i]=( 2*np.abs((x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]))
        / np.sqrt(
        ((x[1]-x[0])**2+(y[1]-y[0])**2)*
        ((x[2]-x[0])**2+(y[2]-y[0])**2)*
        ((x[2]-x[1])**2+(y[2]-y[1])**2)) )

    return K

def get_periodic_table():


	# Set the Peridi
	pt_table = { 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 
	'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 
	'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 
	'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46,
	'ag': 47, 'cd': 48, 'in': 49, 'sn': 50, 'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56, 'la': 57, 'ce': 58, 
	'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64, 'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70,
	'lu': 71, 'hf': 72, 'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80, 'tl': 81, 'pb': 82, 
	'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88, 'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93, 'pu': 94,
    'am': 95, 'cm': 96, 'bk': 97, 'cf': 98, 'es': 99, 'fm': 100, 'md': 101, 'no': 102, 'lr': 103, 'rf': 104}

	return pt_table
