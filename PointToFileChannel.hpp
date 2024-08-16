#pragma once

#include <ns3/log-macros-enabled.h>
#include <ns3/point-to-point-channel.h>

#include <filesystem>
#include <thread>

class PointToFileChannel : public ns3::PointToPointChannel {

  // reuse existing constructor of base class
  PointToFileChannel(std::filesystem::path const &conn_file)
      : m_conn_out(conn_file / "_out"), m_conn_in(conn_file / "_in"),
        ns3::PointToPointChannel() {}

  ns3::Ptr<ns3::Packet> waitForIncomingData() {
    while (!checkDataAvailable()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return readDataFromFile();
  }

  bool checkDataAvailable() {
    if (!std::filesystem::exists(m_conn_in)) {
      return false;
    }
    return true;
  }

  ns3::Ptr<ns3::Packet> readDataFromFile() {
    std::ifstream ifs{m_conn_in};
    if (!ifs) {
      NS_LOG_ERROR("couldn't read from file: " << m_conn_in.string());
      return nullptr;
    }

    auto size = std::filesystem::file_size(m_conn_in);
    std::vector<char> bytes(size);
    ifs.read(bytes.data(), size);

    auto packet = ns3::Create<ns3::Packet>(bytes.data(), size);
    return packet;
  }

  virtual bool TransmitStart(ns3::Ptr<const ns3::Packet> p,
                             ns3::Ptr<ns3::PointToPointNetDevice> src,
                             ns3::Time txTime) override {
    namespace fs = std::filesystem;

    if (!fs::exists(m_conn_out)) {
      fs::remove(m_conn_out);
    }

    std::ofstream ofs{m_conn_out};
    if (!ofs) {
      NS_LOG_ERROR("Couldn't write to file " << m_conn_out.string());
      return false;
    }

    p->CopyData(&ofs, p->GetSize());
  }

private:
  std::filesystem::path m_conn_out;
  std::filesystem::path m_conn_in;
};