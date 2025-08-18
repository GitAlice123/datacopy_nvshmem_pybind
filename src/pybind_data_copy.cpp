#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "double_buffer_manager.h" 


std::vector<uint8_t> get_unique_id();
int init_nvshmem(const std::vector<uint8_t>& root_unique_id, int rank, int world_size);

namespace py = pybind11;

PYBIND11_MODULE(pydatacopy, m) {
    py::class_<DoubleBufferManager>(m, "DoubleBufferManager")
        .def(py::init<>())
        .def("init", &DoubleBufferManager::init, "Initialize double buffer")
        .def("cleanup", &DoubleBufferManager::cleanup, "Cleanup double buffer")
        .def("test_bandwidth", &DoubleBufferManager::test_bandwidth, "Test bandwidth");
    
    m.def("get_unique_id", &get_unique_id, "Generate a unique ID for NVSHMEM initialization");
    m.def("init_nvshmem", &init_nvshmem, "Initialize NVSHMEM",
          py::arg("root_unique_id"), py::arg("rank"), py::arg("world_size"));
}