//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "ctvlib.hpp"
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <random>

using namespace Eigen;
using namespace std;
namespace py = pybind11;

ctvlib::ctvlib(int Nx, int Ny)
{
    nx = Nx;
    ny = Ny;
    nTotal = nx * ny;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++-OMP implementation of Isentropic TV Measurement (2D case)
// Input Parameters:
// Noisy image/volume

// Output:
// TV Measurement

float ctvlib::tv(MatrixXf recon)
{
    float eps = 1e-8;
    recon.resize(nx,ny);
    MatrixXf tv_recon(nx,ny);
    
    #pragma omp parallel for
    for (int i = 0; i < nx; i++)
    {
        int ip = (i+1)%nx;
        for (int j = 0; j < ny; j++)
        {
            int jp = (j+1)%ny;
            tv_recon(i,j) = sqrt(eps + ( recon(i,j) - recon(ip,j) ) * ( recon(i,j) - recon(ip,j) )
                                    + ( recon(i,j) - recon(i,jp) ) * ( recon(i,j) - recon(i,jp) ) );
        }
    }
    
    return tv_recon.sum();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++-OMP implementation of Subgradient Descent-TV [1] denoising/regularization model (2D case)
// Input Parameters:
// Noisy image/volume
// lambdaPar - regularization parameter
// Number of iterations

// Output:
// Filtered/regularized image/volume

// This function is based on the paper by
// [1] Emil Sidky, Chien-Min Kao, and Xiaochuan Pan, "Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT"

VectorXf ctvlib::gd_tv(MatrixXf &recon, float dPOCS, int ng)
{
    recon.resize(nx,ny);
    MatrixXf v(nx,ny);
    for (int ll = 0; ll < ng; ll++)
    {
        tv2Dderivative(recon, v);
        v /= v.norm();
        recon.array() -= dPOCS * v.array();
    }
    
    // Non-Negativity
    recon = (recon.array() < 0).select(0, recon);

    recon.resize(nTotal,1);
    VectorXf out = recon;
    return out;
}

void ctvlib::tv2Dderivative(MatrixXf recon, MatrixXf &tv_recon)
{
    float eps = 1e-8;
    int i, j, ip, im, jp, jm;
    float v1n, v1d, v2n, v2d, v3n, v3d;
    
    
    #pragma omp parallel for
    for (i = 0; i < nx; i++)
    {
        ip = (i+1) % nx;
        im = (i-1+nx) % nx;
        
        for (j = 0; j < ny; j++)
        {
            jp = (j+1) % ny;
            jm = (j-1+ny) % ny;
                
            v1n = 2.0 * ( recon(i,j) - recon(im,j) ) + 2.0 * ( recon(i,j) - recon(i,jm) );
            v1d = sqrt(eps + ( recon(i,j) - recon(im,j) ) * ( recon(i,j) - recon(im,j) )
                              +  ( recon(i,j) - recon(i,jm) ) * ( recon(i,j) - recon(i,jm) ) );
            
            v2n = 2.0 * ( recon(ip,j) - recon(i,j) );
            v2d = sqrt(eps + ( recon(ip,j) - recon(i,j) ) * ( recon(ip,j) - recon(i,j) )
                             +  ( recon(ip,j) - recon(ip,jm) ) * ( recon(ip,j) - recon(ip,jm) ) );
            
            v3n =  2.0 * ( recon(i,jp) - recon(i,j) );
            v3d = sqrt(eps + ( recon(i,jp) - recon(i,j) ) * ( recon(i,jp) - recon(i,j) )
                             +  ( recon(i,jp) - recon(im,jp) ) * ( recon(i,jp) - recon(im,jp) ) );
            
            tv_recon(i,j) = v1n/v1d - v2n/v2d - v3n/v3d;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++-OMP implementation of FGP-TV [2] denoising/regularization model (2D case)
// Input Parameters:
// Noisy image/volume
// lambdaPar - regularization parameter
// Number of iterations

// Output:
// Filtered/regularized image/volume

// This function is based on the paper by
// [2] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"


VectorXf ctvlib::fgp_tv(Eigen::MatrixXf &Input, float lambda, int ng)
{
    Input.resize(nx,ny);
    MatrixXf Px = MatrixXf::Zero(nx,ny);
    MatrixXf Py = MatrixXf::Zero(nx,ny);
    MatrixXf Output = MatrixXf::Zero(nx,ny);
    
    float multip = (1.0f/(8.0f*lambda));
    
    // Main Loop
    for (int ll = 0; ll < ng; ll++)
    {
        // Compute Gradient of Objective Function
        obj_func2D(Input, Output, Px, Py, lambda);

        // Take step towards minus of gradient
        grad_func2D(Output, Px, Py, multip);
        
        // Projection Step
        proj_func2D_iso_kernel(Px, Py); 
    }
    
    // Positivity
    Output = (Output.array() < 0).select(0, Output);
    
    // Figure out how to do more efficiently..
    Output.resize(nTotal,1);
    VectorXf out = Output;
    return out;
}

void ctvlib::obj_func2D(MatrixXf &Input, MatrixXf &Output, MatrixXf &Px, MatrixXf &Py, float lambda)
{
    float val1, val2;
    int i, j;
    
    #pragma omp parallel for shared(Input, Output, Px, Py, multip) private(i, j, val1, val2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            // Boundary Conditions
            if ( i == 0 ) {val1 = 0;} else {val1 = Px(i-1, j);}
            if ( j == 0 ) {val2 = 0;} else {val2 = Py(i, j-1);}
            
            Output(i,j) = Input(i,j) - lambda * (Px(i,j) + Py(i,j) - val1 - val2);
        }
    }
}

void ctvlib::grad_func2D(MatrixXf &Im, MatrixXf &Px, MatrixXf &Py, float multip)
{
    float val1, val2;
    int i, j;
    
    #pragma omp parallel for shared(Im, Px, Py, multip) private(i, j, val1, val2)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            if ( i == nx - 1 ) {val1 = 0;} else {val1 = Im(i,j) - Im(i+1, j);}
            if ( j == ny - 1 ) {val2 = 0;} else {val2 = Im(i,j) - Im(i, j+1);}
            
            Px(i,j) += multip * val1;
            Py(i,j) += multip * val2;
        }
    }
}

void ctvlib::proj_func2D_iso_kernel(MatrixXf &Px, MatrixXf &Py)
{
    float denom;
    int i, j;
    
    #pragma omp parallel for shared(Px,Py) private(i,j,denom)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            denom = Px(i,j) * Px(i,j) + Py(i,j) * Py(i,j);
            
            if (denom > 1.0f ) {
                Px(i,j) /= sqrt(denom);
                Py(i,j) /= sqrt(denom);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C++-OMP implementation of Chambolle-TV [3] denoising/regularization model (2D case)
// Input Parameters:
// Noisy image/volume
// lambdaPar - regularization parameter
// Number of iterations

// Output:
// Filtered/regularized image/volume

// This function is based on the paper by
// [3] Joan Duran, Bartomeu Coll and Catalina Sbert, "Chambolle's Projection Algorithm for Total Variation Denoising"


VectorXf ctvlib::chambolle_tv(MatrixXf &img, float lambda, int ng)
{
    MatrixXf gdvX = MatrixXf::Zero(nx,ny), Px = MatrixXf::Zero(nx,ny);
    MatrixXf gdvY = MatrixXf::Zero(nx,ny), Py = MatrixXf::Zero(nx,ny), divP = MatrixXf::Zero(nx,ny);
    img.resize(nx,ny);
    float multip = (float) 1 / 8;
    
    // Main Loop
    for (int ll=0; ll < ng; ll++)
    {
        // Divergence - (div p - lambda f)
        div(img, Px, Py, divP, lambda, false);
        
        // Gradient - D (div p - lambda f)
        grad(divP, gdvX, gdvY, multip);
        
        // Update Dual Variables (P)
        p_update_isotropic(gdvX, gdvY, Px, Py, multip);
    }
    
    // Divergence (div p)
    div(img, Px, Py, divP, lambda, true);
    
    // Image Update (u = f - 1/lambda div p)
    chambolle_update(img, divP, lambda);
    
    // Positivity
    img = (img.array() < 0).select(0, img);
    
    // Figure out how to do more efficiently..
    img.resize(nTotal,1);
    VectorXf out = img;
    
    return out;
}

// div(img, Px, Py, divP, lambda, false) -- div(img, Px, Py, divP, lambda, true)
void ctvlib::div(MatrixXf &img, MatrixXf &px, MatrixXf &py, MatrixXf &divP, float lambda, bool finalUpdate)
{
    float val1, val2;
    int i, j;

    #pragma omp parallel for shared(divP, img, px, py) private(i,j,val1,val2)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            
            if (i == 0)           { val1 = px(i,j); }
            else if (i == nx - 1) { val1 = -px(i-1,j); }
            else                  { val1 = px(i,j) - px(i-1,j); }

            if (j == 0)           { val2 = py(i,j); }
            else if (j == ny - 1) { val2 = -py(i,j-1); }
            else                  { val2 = py(i,j) - py(i,j-1); }
            
            if (finalUpdate == false) { divP(i,j) = val1 + val2 - img(i,j) / lambda; }
            else { divP(i,j) = val1 + val2; }
        }
    }
}


// grad(divP, gdvX, gdvY, multip);
void ctvlib::grad(MatrixXf &u, MatrixXf &dx, MatrixXf &dy, float multip)
{
    int i, j;
    
    #pragma omp parallel for shared(u,dx,dy) private(i,j)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            
            // Derivitive in X-Direction
            if (i < nx - 1) { dx(i,j) = u(i+1,j) - u(i,j) ; }
            else            { dx(i,j) = 0; }
            
            // Derivitive in Y-Direction
            if (j < ny - 1) { dy(i,j) = u(i,j+1) - u(i,j) ; }
            else            { dy(i,j) = 0; }
        }
    }
}
                
// p_update(gdvX, gdvY, Px, Py, multip)
void ctvlib::p_update_anisotropic(MatrixXf &gdvX, MatrixXf &gdvY, MatrixXf &px, MatrixXf &py, float multip)
{
    int i, j;

    #pragma omp parallel for shared(px,py,gdvX,gdvY) private(i,j)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            px(i,j) = ( px(i,j) + multip * gdvX(i,j) ) / ( 1 + multip * abs(gdvX(i,j)) );
            py(i,j) = ( py(i,j) + multip * gdvY(i,j) ) / ( 1 + multip * abs(gdvY(i,j)) );
        }
    }
}

// p_update(gdvX, gdvY, Px, Py, multip)
void ctvlib::p_update_isotropic(MatrixXf &gdvX, MatrixXf &gdvY, MatrixXf &px, MatrixXf &py, float multip)
{
    int i, j;
    
    #pragma omp parallel for shared(px,py,gdvX,gdvY) private(i,j)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            
            float d = sqrt(gdvX(i,j) * gdvX(i,j) + gdvY(i,j) * gdvY(i,j));
            
            px(i,j) = ( px(i,j) + multip * gdvX(i,j) ) / ( 1 + multip * d );
            py(i,j) = ( py(i,j) + multip * gdvY(i,j) ) / ( 1 + multip * d );
        }
    }
}

// chambolle_update(img, divP, lambda)
void ctvlib::chambolle_update(MatrixXf &img, MatrixXf &divP, float lambda)
{
    int i, j;
    
    #pragma omp parallel for shared(img, divP) private(i,j)
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            img(i,j) -= lambda * divP(i,j);
        }
    }
}
