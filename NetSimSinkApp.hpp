#pragma once

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <map>
#include <sstream>

using namespace ns3;

class SinkApp : public Application {
public:
  SinkApp(std::vector<Ptr<Socket>> socket)
      : m_running(false), m_sockets(socket) {};
  virtual ~SinkApp() {};

  virtual void StartApplication() override {
    m_running = true;

    // Listen for incoming connections
    for (auto const &socket : m_sockets) {
      socket->Listen();
      socket->SetAcceptCallback(
          MakeNullCallback<bool, Ptr<Socket>, const Address &>(),
          MakeCallback(&SinkApp::onAccept, this));
    }
  };
  virtual void StopApplication() override {

    m_running = false;
    for (auto &socket : m_sockets) {
      if (socket) {
        socket->Close();
      }
    }
  };

  auto getMeasurements() -> std::vector<std::vector<uint8_t>> {
    return m_measurements;
  }

private:
  void onAccept(Ptr<Socket> s, const Address &from) {
    s->SetRecvCallback(MakeCallback(&SinkApp::onReceive, this));
  }

  void onReceive(Ptr<Socket> socket) {
    auto peerAddress = getPeerAddress(socket);
    std::vector<uint8_t> buffer;
    Ptr<Packet> packet;

    while ((packet = socket->Recv())) {
      if (packet->GetSize() == 0) { // EOF
        break;
      }

      // copy packet to buffer
      auto offset = buffer.size();
      buffer.resize(offset + packet->GetSize());
      packet->CopyData(buffer.data() + offset, packet->GetSize());
    }

    NS_LOG_UNCOND("Received" << buffer.size() << " bytes from " << peerAddress);

    // save the received data in a map
    m_measurements.push_back(buffer);
  }

  std::string getPeerAddress(Ptr<Socket> socket) {
    Address addr;
    int success = socket->GetPeerName(addr);
    std::stringstream ss;
    ss << addr;
    return ss.str();
  }
  std::vector<std::vector<uint8_t>> m_measurements;
  bool m_running;
  std::vector<Ptr<Socket>> m_sockets;
};