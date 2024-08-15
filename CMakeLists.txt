cmake_minimum_required(VERSION 3.10)

project(netsim)

file(COPY_FILE ${CMAKE_SOURCE_DIR}/src/NetSim.cc ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc ONLY_IF_DIFFERENT)

add_subdirectory(ext/ns3)

target_compile_features(scratch_NetSim PUBLIC cxx_std_20)