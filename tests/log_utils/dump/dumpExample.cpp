// Copyright (c) 2020, SenseTime.
#include <pybind11/pybind11.h>

namespace py = pybind11;

int dump(int a = 0) {
    return *(reinterpret_cast<int *>(a));
}

PYBIND11_MODULE(dump_example, m) {
    m.def("dump", &dump, py::arg("a"));
}
