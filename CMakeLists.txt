cmake_minimum_required(VERSION 3.10)

project(netsim)

# file(COPY_FILE ${CMAKE_SOURCE_DIR}/NetSim.cc ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc ONLY_IF_DIFFERENT)
set(NS3_EXAMPLES ON) # build examples as well
set(NS3_LOG_ENABLE ON)

add_custom_command(
    OUTPUT
    ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc
    DEPENDS
    ${CMAKE_SOURCE_DIR}/NetSim.cc
    COMMAND
        cmake -E copy_if_different ${CMAKE_SOURCE_DIR}/NetSim.cc ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc
    )
add_custom_target(netsim_src DEPENDS ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc)

add_subdirectory(ext/ns3)

target_compile_features(scratch_NetSim PUBLIC cxx_std_20)
add_dependencies(scratch_NetSim netsim_src)