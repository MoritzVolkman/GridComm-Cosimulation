#pragma once

#include <ns3/point-to-point-net-device.h>

#include <filesystem>

class PointToPointFileDevice : public ns3::PointToPointNetDevice {

  PointToPointFileDevice(std::string const &conn_name)
      : m_conn_dir(conn_name), m_conn_out(m_conn_dir / "out"),
        m_conn_in(m_conn_dir / "in") {}

  // checks for any messages available in the connection file
  void checkIncoming() {
    namespace fs = std::filesystem;
    if (!fs::exists(m_conn_in)) {
    } else {
      // new data is available
      std::ifstream ifs{m_conn_in};
      if (ifs) {
        auto size = std::filesystem::file_size(m_conn_in);
        std::vector<char> packet(size);
        ifs.read(packet.data(), size);
        
      } else {
        NS_LOG_ERROR("Cannot read from file" << m_conn_in.string());
        return;
      }

      ifs.close();
      fs::remove(m_conn_in); // remove file to mark completion receiving
    }
  }

  bool Send(ns3::Ptr<ns3::Packet> packet, const ns3::Address &dest,
            uint16_t protocolNumber) override {
    namespace fs = std::filesystem;
    // the file directory containing all the connection's files
    auto conn_dir = fs::path(m_conn_name);
    auto conn_out =
        conn_dir /
        "out"; // a file representing messages going out to this device
    auto conn_in =
        conn_dir /
        "in"; // a file representing messages arriving from this device

    // write packet to file and comm log
    fs::create_directories(m_conn_name);
    if (fs::exists(m_conn_name))
      fs::remove();

    return true;
  }

private:
  std::filesystem::path m_conn_dir;
  std::filesystem::path m_conn_in;
  std::filesystem::path m_conn_out;
};