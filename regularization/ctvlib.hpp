
//
//  ctvlib.hpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef ctvlib_hpp
#define ctvlib_hpp

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

class ctvlib 
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

public: 

	// Initializes Measurement Matrix. 
	ctvlib(int Nrow, int Ncol);
    
    int nx, ny, nTotal;
    
    // Measure Isotropic TV
    float tv(Eigen::MatrixXf input);

    // Functions for TV Gradient Descent
    Eigen::VectorXf gd_tv(Eigen::MatrixXf &recon, float dPOCS, int ng);
    void tv2Dderivative(Eigen::MatrixXf recon, Eigen::MatrixXf &tv_recon);

    // Functions for Fast Gradient Projection Method
    Eigen::VectorXf fgp_tv(Eigen::MatrixXf &recon, float eps, int ng);
    void obj_func2D(Eigen::MatrixXf &Input, Eigen::MatrixXf &Output, Eigen::MatrixXf &Px, Eigen::MatrixXf &Py, float lambda);
    void grad_func2D(Eigen::MatrixXf &recon, Eigen::MatrixXf &Px, Eigen::MatrixXf &Py, float multip);
    void proj_func2D_iso_kernel(Eigen::MatrixXf &Px, Eigen::MatrixXf &Py);
    
    // Functions for Channel Independent Chambolle Projection Method
    Eigen::VectorXf chambolle_tv(Eigen::MatrixXf &img, float lambda, int ng);
    void div(Eigen::MatrixXf &img, Eigen::MatrixXf &px, Eigen::MatrixXf &py, Eigen::MatrixXf &divP, float lambda, bool finalUpdate);
    void grad(Eigen::MatrixXf &u, Eigen::MatrixXf &dx, Eigen::MatrixXf &dy, float multip);
    void p_update_anisotropic(Eigen::MatrixXf &gdvX, Eigen::MatrixXf &gdvY, Eigen::MatrixXf &px, Eigen::MatrixXf &py, float multip);
    void p_update_isotropic(Eigen::MatrixXf &gdvX, Eigen::MatrixXf &gdvY, Eigen::MatrixXf &px, Eigen::MatrixXf &py, float multip);
    void chambolle_update(Eigen::MatrixXf &img, Eigen::MatrixXf &divP, float lambda);
    
};

#endif /* ctvclib_hpp */
