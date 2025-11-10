#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "point_cloud_processor.h"

namespace py = pybind11;

PYBIND11_MODULE(perception_cpp, m) {
    m.doc() = "C++ perception library for Text2Wheel - handles all navigation logic";

    py::class_<Point>(m, "Point")
        .def(py::init<double, double>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);

    py::class_<NavigationCommand>(m, "NavigationCommand")
        .def(py::init<>())
        .def_readwrite("linear_velocity", &NavigationCommand::linear_velocity)
        .def_readwrite("angular_velocity", &NavigationCommand::angular_velocity);

    py::class_<PointCloudProcessor>(m, "PointCloudProcessor")
        .def(py::init<double>())
        .def("lidar_to_point_cloud", &PointCloudProcessor::lidar_to_point_cloud)
        .def("remove_noise", &PointCloudProcessor::remove_noise)
        .def("get_avoidance_vector", &PointCloudProcessor::get_avoidance_vector)
        .def("compute_navigation_command", &PointCloudProcessor::compute_navigation_command);
}
