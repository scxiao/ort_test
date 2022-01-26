cmake_minimum_required(VERSION 3.5)
project (Test_ORT)

set (ORT_FOLDER /code/onnxruntime)
set (ORT_LIBS ${ORT_FOLDER}/build/Linux/Release)
#set (ORT_LIBS ${ORT_FOLDER}/build/Linux/Debug)
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_VERBOSE_MAKEFILE ON)

include_directories(${ORT_FOLDER}/include/onnxruntime
		    ${ORT_FOLDER}/onnxruntime/
                    ${ORT_FOLDER}/include/onnxruntime/core/session)
                    
link_directories(${ORT_LIBS})

file(GLOB util_SRC "*.cpp")
file(GLOB test_examples test_char_rnn.cpp)
set(test_examples run_onnx.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} onnxruntime)
endforeach(filepath_name)

