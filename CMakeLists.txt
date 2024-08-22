cmake_minimum_required(VERSION 3.10)

project(netsim)

# Surpress cmake developer warnings for fetched libs
cmake_policy(SET CMP0135 NEW)

# Download JSON library
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# Download the boost library (for physical networking)
set(BOOST_INCLUDE_LIBRARIES system asio headers)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.86.0
)
FetchContent_MakeAvailable(Boost)

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
  Boost::headers
  Boost::asio
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

