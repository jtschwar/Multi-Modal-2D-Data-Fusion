#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ctvlib.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ctvlib, m)
{
    m.doc() = "Fused Multi-Modal Regularization Library";
    py::class_<ctvlib> ctvlib(m, "ctvlib");
    ctvlib.def(py::init<int,int>());
    ctvlib.def("tv", &ctvlib::tv, "TV Measurement");
    ctvlib.def("gd_tv", &ctvlib::gd_tv, "TV Gradient Descent");
    ctvlib.def("fgp_tv", &ctvlib::fgp_tv, "Fast Gradient Projection Method");
    ctvlib.def("chambolle_tv", &ctvlib::chambolle_tv, "Chambolle Projection Algorithm");
}
