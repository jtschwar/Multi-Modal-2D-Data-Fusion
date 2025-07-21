import numpy as np

# Class for Gradient Projection Method for Total Variation Denoising 
class tvlib:

    def __init__(self, nx, ny):
        self.nx = nx; self.ny = ny

    # Projection Kernel 
    def P_p(self, p,q, kernel='anisotropic'):

        # p = np.hstack([np.zeros(nx), p])
        # q = np.hstack([q, np.zeros(ny)])

        if kernel == 'isotropic':
            p = p / np.maximum(1, np.sqrt(p**2 + q**2) ) 
            q = q / np.maximum(1, np.sqrt(p**2 + q**2) ) 
        else: 
            p = p / np.maximum(1, np.abs(p))
            q = q / np.maximum(1, np.abs(q))

        return p, q

    # Projection onto Convex Set
    def P_c(self, x):
        return np.clip(x, 0, None)

    # Linear Operator
    def L(self, p, q):

        num_cols_p = p.shape[1]; num_rows_q = q.shape[0]
        
        p[0,:] = 0; p[-1,:] = 0
        q = np.hstack((q, np.zeros((num_rows_q, 1))))
        
        q[:,0] = 0
        p = np.vstack((p, np.zeros((1,num_cols_p))))
        
        #Produces one matrix that is m x n
        return p + q - np.roll(p, -1, axis=0) - np.roll(q, -1, axis=1)

    # Transpose of Linear Operator 
    def L_t(self, x , old_p, old_q, coeff=1):

        m,n = x.shape
        new_p = x[:-1,:] - np.roll(x[:-1,:], 1, axis=0)
        new_q = x[:,:-1] - np.roll(x[:,:-1], 1, axis=1)

        return old_p + coeff * new_p, old_q + coeff * new_q     
                

    # Python Implementation of FGP-TV [1] denoising / regularization (2D Case)
    # This function is based on the paper by
    # [2] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
    def fgp_tv(self, input, lambdaTV = 1e2, nIter = 25, momentum=False):  
        
        (nx,ny) = input.shape

        output = input.copy()
        Px = np.zeros([nx-1,ny]); Py = np.zeros([nx,ny-1])

        if momentum: Rx = Px.copy; Ry = Py.copy; t0 = 1 

        # Main Loop 
        coeff = 1 / (8 * lambdaTV)
        for i in range(nIter):
            
            output = self.P_c( input - lambdaTV * self.L(Px,Py) ) 
            (Px, Py) = self.L_t(output, Px, Py, coeff=coeff)
            (Px, Py) = self.P_p(Px, Py)

            # if momentum:
            #     t = 0.5 * ( 1 + np.sqrt(1 + 4*(t0**2)) )
            #     r,s = ( p_1 + ( ((t-1)/t0)*(p_1 - p) ), q_1 + ( ((t-1)/t0)*(q_1 - q) ) )
            #     p = p_1; q = q_1; t = t0

        return self.P_c( input - lambdaTV * self.L(Px,Py) )


    def tv(self, input, kernel='isotropic'):
          
        if kernel == 'isotropic':  
            deltaX = np.diff(input,axis=0)**2
            deltaY = np.diff(input,axis=1)**2 
            tv = np.sqrt(np.sum(deltaX) + np.sum(deltaY))
        return tv
