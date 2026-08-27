# Tarea-Prog3
DESAROLLO:
El proyecto fue realizado individualmente por Nicolas Lara Alvarez.

CODIGO:
El codigo contiene algunos comentarios para facilitar la lectura, especialmente en Tensor.hpp. esto debido a la cantidad de operadores que contiene y los cuales 
fueron usados por Tensor.cpp.

EJECUCION:
Para la ejecucion se colocó el CMakeLists y tambien por este medio:

cmake_minimum_required(VERSION 4.3)
project(Tarea)

set(CMAKE_CXX_STANDARD 20)

add_compile_options(-Wall -Wextra)

add_executable(main
        Tensor.hpp
        Tensor.cpp
        main.cpp
)

