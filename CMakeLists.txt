cmake_minimum_required(VERSION 3.0.0)
project(faceanglecheck VERSION 0.1.0)


#Add java and jni
if(CMAKE_TOOLCHAIN_FILE)
    set(JAVA_AWT_LIBRARY NotNeeded)
    set(JAVA_JVM_LIBRARY NotNeeded)
    set(JAVA_INCLUDE_PATH2 NotNeeded)
    set(JAVA_INCLUDE_PATH NotNeeded)
    set(JAVA_AWT_INCLUDE_PATH NotNeeded)
endif()
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
find_package(OpenCV REQUIRED)
include(UseJava)

#compile against which c++
set (CMAKE_CXX_STANDARD 14)

include(CTest)
enable_testing()


#Include jni directories
include_directories(${JNI_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


#Add source to the general files
#file (GLOB SOURCE_FILE src/*.h src/*.cpp)
file (GLOB_RECURSE SOURCE_FILE src/*.h src/*.cpp)


add_library(samplefacecrop SHARED ${SOURCE_FILE})

set(LIBS ${JNI_LIBRARIES} ${OpenCV_LIBS}) 

#Build for Mac OS.
if(NOT CMAKE_TOOLCHAIN_FILE)
    message("--Generating both macos library and executables ")
    target_link_libraries(samplefacecrop ${LIBS})
    add_executable(samplefacecropexec samplefacecrop.cpp ${SOURCE_FILE})
    target_link_libraries(samplefacecropexec ${LIBS})
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
