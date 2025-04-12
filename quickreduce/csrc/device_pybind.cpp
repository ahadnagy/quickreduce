#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "device.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &init);
    m.def("get_world_size", &get_world_size);
    m.def("get_rank", &get_rank);
    m.def("get_comm_handle", &get_comm_handle);
    m.def("set_comm_handles", &set_comm_handles);
    m.def("allreduce", &allreduce);
    m.def("fused_gemm_ar", &fused_gemm_ar);
}
