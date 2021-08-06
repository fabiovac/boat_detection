cmake_minimum_required(VERSION 3.19)
project(CV_part)

find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(/usr/local/Cellar/python@3.9/3.9.6/Frameworks/Python.framework/Versions/3.9/Headers/)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} main.cpp BoatDetector.cpp BoatDetector.h PreProcessor.cpp PreProcessor.h)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})