#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "rocksample.h"

namespace py = pybind11;
using namespace std;


py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
  py::buffer_info buf1 = input1.request();
  py::buffer_info buf2 = input2.request();

  if (buf1.size != buf2.size) {
    throw std::runtime_error("Input shapes must match");
  }

  /*  allocate the buffer */
  py::array_t<double> result = py::array_t<double>(buf1.size);


  py::buffer_info buf3 = result.request();

  double *ptr1 = (double *) buf1.ptr,
         *ptr2 = (double *) buf2.ptr,
         *ptr3 = (double *) buf3.ptr;
  int X = buf1.shape[0];
  int Y = buf1.shape[1];

  for (size_t idx = 0; idx < X; idx++) {
    for (size_t idy = 0; idy < Y; idy++) {
      ptr3[idx*Y + idy] = ptr1[idx*Y+ idy] + ptr2[idx*Y+ idy];
    }
  }
 
  // reshape array to match input shape
  result.resize({X,Y});

  return result;
}

PYBIND11_MODULE(rocksample, m) {
    m.doc() = "pybind11 bindings to rocksample simulator"; // optional module docstring
    
    py::class_<ROCKSAMPLE>(m, "Rocksample")
        .def(py::init<int,int,double,bool>())
        .def("getagentx",&ROCKSAMPLE::GetAgentX)
        .def("getagenty",&ROCKSAMPLE::GetAgentY)
        .def("step",&ROCKSAMPLE::Step)
        .def("get_legalactions",&ROCKSAMPLE::GenerateLegal)
        .def("rollout",&ROCKSAMPLE::Rollout)
        .def("get_state",&ROCKSAMPLE::get_state)
        .def("set_state",&ROCKSAMPLE::set_state);
    
    m.def("add_arrays", &add_arrays, "Add two NumPy arrays"); 
    //m.def("get_state", &get_state, "get a numpy array");  
}


