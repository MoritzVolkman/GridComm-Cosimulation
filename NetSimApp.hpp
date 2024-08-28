#pragma once

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"

using namespace ns3;

class NetSimApp : public Application {
public:
  NetSimApp()
      : m_socket(0), m_peer(), m_packetSize(0), m_nPackets(0), m_dataRate(0),
        m_sendEvent(), m_running(false), m_packetsSent(0) {}

  virtual ~NetSimApp() { m_socket = 0; }

  void LoadJsonData(std::string path, std::vector<std::string> jsonData);
  void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
             DataRate dataRate, std::string jsonData) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = jsonData.size() / packetSize +
                 (jsonData.size() % packetSize > 0 ? 1 : 0);
    m_dataRate = dataRate;
    m_jsonData = {jsonData.begin(), jsonData.end()};
  }

private:
  virtual void StartApplication(void) {
    m_running = true;
    m_packetsSent = 0;

    // Connect the socket
    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
  }

  virtual void StopApplication() {
    m_running = false;

    if (m_sendEvent.IsPending()) {
      Simulator::Cancel(m_sendEvent);
    }

    if (m_socket) {
      m_socket->Close();
    }
  }

  void ScheduleTx() {
    if (m_running) {
      Time tNext(Seconds(m_packetSize * 8 /
                         static_cast<double>(m_dataRate.GetBitRate())));
      m_sendEvent = Simulator::Schedule(tNext, &NetSimApp::SendPacket, this);
    }
  }

  void SendPacket() {
    auto packetOffset = m_packetSize * m_packetsSent;
    auto packetSize = m_packetSize;
    auto remainingBytes =
        std::distance(m_jsonData.begin() + packetOffset, m_jsonData.end());
    if (remainingBytes < m_packetSize) {
      packetSize = remainingBytes;
    }

    Ptr<Packet> packet =
        Create<Packet>(m_jsonData.data() + packetOffset, remainingBytes);

    m_socket->Send(packet);

    if (++m_packetsSent < m_nPackets) {
      ScheduleTx();
    }
  }

  Ptr<Socket> m_socket;
  Address m_peer;
  uint32_t m_packetSize;
  uint32_t m_nPackets;
  DataRate m_dataRate;
  std::vector<uint8_t> m_jsonData;
  EventId m_sendEvent;
  bool m_running;
  uint32_t m_packetsSent;
};
