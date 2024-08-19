cmake_minimum_required(VERSION 3.10)

# Allow linking libraries to targets of subdirectories (e.g. json to scratch targets)
if(POLICY CMP0079 )
  cmake_policy(SET CMP0079 NEW)
endif()

project(netsim)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

file(COPY_FILE ${CMAKE_SOURCE_DIR}/NetSim.cc ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/NetSim.cc ONLY_IF_DIFFERENT)
file(COPY_FILE ${CMAKE_SOURCE_DIR}/test.cc ${CMAKE_SOURCE_DIR}/ext/ns3/scratch/test.cc ONLY_IF_DIFFERENT)

add_subdirectory(ext/ns3)

set(PCAP_FILES_PATH "${CMAKE_SOURCE_DIR}/PCAP/NetSim")
configure_file(${CMAKE_SOURCE_DIR}/pcap.hpp_in ${CMAKE_BINARY_DIR}/pcap/pcap.hpp)
target_include_directories(scratch_NetSim PUBLIC "${CMAKE_BINARY_DIR}/pcap")
target_include_directories(scratch_test PUBLIC "${CMAKE_BINARY_DIR}/pcap")

target_compile_features(scratch_NetSim PUBLIC cxx_std_20)
target_link_libraries(scratch_NetSim nlohmann_json::nlohmann_json)
target_compile_features(scratch_test PUBLIC cxx_std_20)
target_link_libraries(scratch_test nlohmann_json::nlohmann_json)
