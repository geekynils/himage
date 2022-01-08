cmake_minimum_required(VERSION 3.7)

project(himage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sanitize w/o xcode
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined")
# set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address,undefined")

set(STB_HEADERS CACHE FILEPATH "")

include_directories(${STB_HEADERS})

set(SRC
    image.cpp
    image_ops.cpp
    canny.cpp
    vec.cpp
)

set(HEADER
    image.h
    image_ops.h
    canny.h
    vec.h
)

add_executable(tests test.cpp ${SRC} ${HEADER})
add_executable(find_edges main.cpp ${SRC} ${HEADER})