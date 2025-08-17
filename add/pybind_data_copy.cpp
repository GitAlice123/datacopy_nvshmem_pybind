#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

void cuda_cross_device_test(int rank, int world_size, const std::vector<uint8_t>& root_unique_id);
std::vector<uint8_t> get_unique_id();

namespace py = pybind11;

PYBIND11_MODULE(pydatacopy, m) {
    m.def("get_unique_id", &get_unique_id, "Generate a unique ID for NVSHMEM initialization");
    
    m.def("cross_device_test", &cuda_cross_device_test, 
          "Perform cross-device communication with NVSHMEM",
          py::arg("rank"), py::arg("world_size"), py::arg("root_unique_id"));
}