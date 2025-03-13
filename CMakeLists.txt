cmake_minimum_required(VERSION 3.22)

if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER dpcpp)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker
    # (needed on Windows only)
    include (CMakeForceCompiler)
    CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP)
    include (Platform/Windows-Clang)
endif()

include(FetchContent)

project(ngbem)

option( USE_KiFMM "Use KiFMM" OFF )
option( USE_FMM3D "Use FMM3D" OFF )

option( USE_IFGF "Use IFGF" OFF )

include(ngsolve_addon.cmake)

SET(SOURCE_FILES src/ngbem.cpp src/python_bem.cpp
  src/hmat.cpp src/intrules.cpp src/test_compression.cpp )
Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
string(APPEND CMAKE_CXX_FLAGS_RELEASE "-march=native  -Ofast -funroll-loops  -UUSE_NGSOLVE -flto=auto -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86)# -fsycl-host-compiler=g++ -std=c++20")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2  -flto=auto -O")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O0  -flto=auto")
#add_definitions(-DEIGEN_FAST_MATH=1 -DEIGEN_DONT_PARALLELIZE   )

add_ngsolve_addon(_ngbem ${SOURCE_FILES})
  set_property(TARGET _ngbem PROPERTY CXX_STANDARD 20)

if(USE_IFGF)
  




# find_package(unordered_dense CONFIG )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (TBB REQUIRED)

  target_compile_definitions(_ngbem PRIVATE USE_IFGF)
  FetchContent_Declare(
    ifgf
    GIT_REPOSITORY https://github.com/arieder/ifgf.git
    GIT_TAG "main"
    EXCLUDE_FROM_ALL
  )
  FetchContent_Populate(ifgf)

  target_include_directories(_ngbem PRIVATE ${FETCHCONTENT_BASE_DIR}/ifgf-src/)

  find_package (Eigen3 3.3 REQUIRED NO_MODULE)
  find_package (TBB REQUIRED)
  find_package(IntelSYCL REQUIRED)
  set(TARGET_NAME _ngbem)
  
  include_directories(${EIGEN3_INCLUDE_DIR} ${TBB_INCLUDE_DIR})
  SET( CMAKE_CXX_FLAGS_RELEASE  "-Ofast ")
  target_link_libraries(_ngbem  PRIVATE tbb)
  add_sycl_to_target(TARGET ${TARGET_NAME}  SOURCES ${SOURCE_FILES} )

  target_link_libraries (${TARGET_NAME} PRIVATE tbb)

  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Ofast -march=native -UUSE_NGSOLVE -flto=auto")

  
endif(USE_IFGF)

if(USE_KiFMM)
  target_compile_definitions(_ngbem PRIVATE USE_KiFMM)
  FetchContent_Declare(
    kifmm
    GIT_REPOSITORY https://github.com/bempp/kifmm.git
    GIT_TAG "enh/c-abi"
    EXCLUDE_FROM_ALL
  )
  FetchContent_Populate(kifmm)
  add_custom_target(build_kifmm ALL
    COMMAND cargo build -r WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/kifmm-src
    BYPRODUCTS ${FETCHCONTENT_BASE_DIR}/kifmm-src/targets/kifmm_rs.h
    USES_TERMINAL
  )

  target_include_directories(_ngbem PRIVATE ${FETCHCONTENT_BASE_DIR}/kifmm-src/kifmm/include)
  target_link_libraries(_ngbem PRIVATE ${FETCHCONTENT_BASE_DIR}/kifmm-src/target/release/libkifmm.so)
  install(FILES ${FETCHCONTENT_BASE_DIR}/kifmm-src/target/release/libkifmm.so DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/ngbem/)
  add_dependencies(_ngbem build_kifmm)

  set_target_properties(_ngbem PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${ADDON_INSTALL_DIR_PYTHON}/ngbem/"
  )
  
else(USE_KiFMM)
  if(USE_FMM3D)
    target_compile_definitions(_ngbem PRIVATE USE_FMM3D)
    FetchContent_Declare(
      fmm3d
      GIT_REPOSITORY https://github.com/flatironinstitute/FMM3D.git
      GIT_TAG master
      EXCLUDE_FROM_ALL
    )
    FetchContent_Populate(fmm3d)

    set(fmm_lib_path ${fmm3d_SOURCE_DIR}/lib-static/libfmm3d.a)

    add_custom_command(OUTPUT ${fmm_lib_path}
      COMMAND make lib -j 9 WORKING_DIRECTORY ${fmm3d_SOURCE_DIR}
      USES_TERMINAL
    )
    add_custom_target(build_fmm DEPENDS ${fmm_lib_path})

    add_library(fmm STATIC IMPORTED GLOBAL)
    set_target_properties(fmm PROPERTIES IMPORTED_LOCATION ${fmm_lib_path})
    add_dependencies(fmm build_fmm)

    #find_library(fortran_lib NAMES gfortran)
    find_package(OpenMP)

    target_include_directories(_ngbem PRIVATE ${fmm3d_SOURCE_DIR}/fmm3d-src/c)
    target_link_libraries(_ngbem PRIVATE fmm gfortran ${OpenMP_omp_LIBRARY})
  endif(USE_FMM3D)
endif(USE_KiFMM)

install(TARGETS _ngbem DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/ngbem/)

install(FILES src/__init__.py DESTINATION ${ADDON_INSTALL_DIR_PYTHON}/ngbem/)

ngsolve_generate_stub_files(ngbem)
