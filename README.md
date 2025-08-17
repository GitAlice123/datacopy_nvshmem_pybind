# datacopy_nvshmem_pybind
* a datacopy demo using nvshmem, CUDA and pybind
* the focus is communication between GPUs crossing the NET, so pure MPI is not enough
* we use pytorch's dist to initialize connection, and nvshmem to communicate 
