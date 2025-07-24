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
	pt_table = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 
	'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 
	'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 
	'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
	'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 
	'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
	'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 
	'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94,
    'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104}

	return pt_table