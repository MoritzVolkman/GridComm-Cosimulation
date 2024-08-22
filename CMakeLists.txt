cmake_minimum_required(VERSION 3.10)

project(netsim)

# Download JSON library
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# Include build targets of NS3 project
set(NS3_EMU OFF)
set(NS3_TAP OFF)
set(NS3_GSL OFF)
set(NS3_GTK3 OFF)
set(NS3_SQLITE OFF)
set(NS3_EIGEN OFF)
set(NS3_VISUALIZER OFF)
add_subdirectory(ext/ns3)

# Define executable NetSim with related source files
add_executable(NetSim
  ${CMAKE_SOURCE_DIR}/NetSim.cc
)

# Set c++ standard for NetSim target
target_compile_features(NetSim PUBLIC cxx_std_20)

# Define the required dependency libraries for NetSim
target_link_libraries(NetSim
  ns3::libcore
  ns3::libnetwork
  ns3::libinternet
  ns3::libapplications
  ns3::libpoint-to-point
  nlohmann_json::nlohmann_json
)


# Same for test.cc as a playground file
add_executable(test ${CMAKE_SOURCE_DIR}/test.cc)
target_compile_features(test PUBLIC cxx_std_20)
target_link_libraries(test
  ns3::libcore
  ns3::libnetwork
  ns3::libinternet
  ns3::libapplications
  ns3::libpoint-to-point
  nlohmann_json::nlohmann_json
)

