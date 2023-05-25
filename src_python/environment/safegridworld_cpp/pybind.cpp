#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "safegridworld.h"

namespace py = pybind11;
using namespace std;


PYBIND11_MODULE(safegridworld, m) {
    m.doc() = "pybind11 bindings to safegridworld simulator"; // optional module docstring
    
    py::class_<SAFEGRIDWORLD>(m, "Safegridworld")
        .def(py::init<int,int,double,std::string>())
        .def("step",&SAFEGRIDWORLD::Step)
        .def("get_legalactions",&SAFEGRIDWORLD::GenerateLegal)
        .def("rollout",&SAFEGRIDWORLD::Rollout)
        .def("get_state",&SAFEGRIDWORLD::get_state)
        .def("set_state",&SAFEGRIDWORLD::set_state);
    
    //m.def("get_state", &get_state, "get a numpy array");  
}


