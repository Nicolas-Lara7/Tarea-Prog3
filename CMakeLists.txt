cmake_minimum_required(VERSION 4.3)
project(Tarea)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra)

add_executable(main
        Tensor.hpp
        Tensor.cpp
        main.cpp
)