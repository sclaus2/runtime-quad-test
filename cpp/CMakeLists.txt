cmake_minimum_required(VERSION 3.16)
project(demo_create_and_tabulate LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Use Python for detecting Basix when installed using combined build
find_package(Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import basix, os, sys; sys.stdout.write(os.path.dirname(basix.__file__))"
    OUTPUT_VARIABLE BASIX_PY_DIR
    RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (BASIX_PY_DIR)
    message(STATUS "Adding ${BASIX_PY_DIR} to Basix search hints")
  endif()
endif()
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})

find_package(UFCx 0.9 REQUIRED MODULE)

add_executable(${PROJECT_NAME} main.cpp poisson.c)

target_link_libraries(${PROJECT_NAME} Basix::basix)

if(TARGET ufcx::ufcx)
  target_link_libraries(${PROJECT_NAME} PUBLIC ufcx::ufcx)
else()
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${UFCX_INCLUDE_DIRS})
endif()
