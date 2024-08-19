#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/pcap-file-wrapper.h"

#include <pcap.hpp>

using namespace ns3;

// Definiere eine eindeutige Komponente für das Logging
NS_LOG_COMPONENT_DEFINE("TcpExample");

// Custom application to send TCP packets
class MyApp : public Application
{
public:
  MyApp();
  virtual ~MyApp();

  // Setup the application with the necessary parameters
  void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, std::string data);

private:
  virtual void StartApplication(void);
  virtual void StopApplication(void);

  void ScheduleTx(void); // Schedule the packet transmission
  void SendPacket(void); // Send the packet

  Ptr<Socket> m_socket;     // The socket
  Address m_peer;           // The peer address
  uint32_t m_packetSize;    // Size of the packet
  uint32_t m_nPackets;      // Number of packets
  DataRate m_dataRate;      // Data rate
  EventId m_sendEvent;      // Event id for scheduled transmission
  bool m_running;           // Application running flag
  uint32_t m_packetsSent;   // Number of packets sent
  std::string m_data;       // Data to be sent
};

MyApp::MyApp()
    : m_socket(0),
      m_peer(),
      m_packetSize(0),
      m_nPackets(0),
      m_dataRate(0),
      m_sendEvent(),
      m_running(false),
      m_packetsSent(0),
      m_data("")
{
}

MyApp::~MyApp()
{
  m_socket = 0;
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, std::string data)
{
  m_socket = socket;
  m_peer = address;
  m_packetSize = packetSize;
  m_nPackets = nPackets;
  m_dataRate = dataRate;
  m_data = data;
}

void MyApp::StartApplication(void)
{
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind();
  m_socket->Connect(m_peer);
  SendPacket();
}

void MyApp::StopApplication(void)
{
  m_running = false;

  if (m_sendEvent.IsPending())
  {
    Simulator::Cancel(m_sendEvent);
  }

  if (m_socket)
  {
    m_socket->Close();
  }
}

void MyApp::SendPacket(void)
{
  Ptr<Packet> packet = Create<Packet>((uint8_t*)m_data.c_str(), m_data.size());
  m_socket->Send(packet);

  if (++m_packetsSent < m_nPackets)
  {
    ScheduleTx();
  }
}

void MyApp::ScheduleTx(void)
{
  if (m_running)
  {
    Time tNext(Seconds(m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
    m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
  }
}

// Callback function to trace packet sending
void SendPacketTrace(Ptr<const Packet> packet)
{
  NS_LOG_UNCOND("Packet sent at " << Simulator::Now().GetSeconds());
}

// Callback function to trace packet reception, print source address and message content
void ReceivePacketTrace(Ptr<const Packet> packet, const Address &from, const Address &to)
{
  Ipv4Address senderIp = InetSocketAddress::ConvertFrom(from).GetIpv4(); // Quelladresse
  Ipv4Address receiverIp = InetSocketAddress::ConvertFrom(to).GetIpv4(); // Zieladresse
  NS_LOG_UNCOND("Packet received at " << Simulator::Now().GetSeconds() << " from " << senderIp << " to " << receiverIp);

  uint8_t *buffer = new uint8_t[packet->GetSize()];
  packet->CopyData(buffer, packet->GetSize());
  std::string packetContent = std::string((char*)buffer, packet->GetSize());
  delete[] buffer;

  NS_LOG_UNCOND("Packet content: " << packetContent);
}

int main(int argc, char *argv[])
{
  LogComponentEnable("TcpExample", LOG_LEVEL_INFO);

  // JSON-Strings für die Daten, die versendet werden sollen
  std::string jsonData[9];
  for (int i = 0; i < 9; ++i)
  {
    std::ostringstream jsonStream;
    jsonStream << "{\n"
               << "  \"MeasurementData\": {\n"
               << "    \"ActivePower\": 123.45,\n"
               << "    \"ReactivePower\": 67.89,\n"
               << "    \"ApparentPower\": 130.00,\n"
               << "    \"PowerFactor\": 0.95,\n"
               << "    \"Voltage\": 230,\n"
               << "    \"Current\": 5.4,\n"
               << "    \"Frequency\": 50,\n"
               << "    \"EnergyConsumption\": 1500.67,\n"
               << "    \"MaximumDemand\": 120,\n"
               << "    \"MeterStatus\": \"ok\",\n"
               << "    \"EventLogs\": [\n"
               << "      \"Event1: Power Failure at 03:00\",\n"
               << "      \"Event2: Power Restoration at 03:10\"\n"
               << "    ]\n"
               << "  },\n"
               << "  \"UserInformation\": {\n"
               << "    \"ConsumerID\": " << i << ",\n"
               << "    \"ContractAccountNumber\": \"CA7891011\",\n"
               << "    \"MeterPointAdministrationNumber\": \"MPAN987654\",\n"
               << "    \"AggregatorID\": \"AG87654321\",\n"
               << "    \"SupplierID\": \"SP12345678\",\n"
               << "    \"DirectMarketerID\": \"DM12345678\"\n"
               << "  }\n"
               << "}";
    jsonData[i] = jsonStream.str();
  }

  // Create 10 nodes
  NodeContainer nodes;
  nodes.Create(10);

  // Set up point-to-point link attributes
  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
  pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

  InternetStackHelper stack;
  stack.Install(nodes); // Install internet stack on nodes

  // Assign unique subnets for each connection
  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> interfaces(9);

  // Create point-to-point connections between each client node and the sink node
  for (uint32_t i = 0; i < 9; ++i)
  {
    NetDeviceContainer devices = pointToPoint.Install(NodeContainer(nodes.Get(i), nodes.Get(9)));
    std::ostringstream subnet;
    subnet << "10.1." << i + 1 << ".0"; // Unique subnet for each connection
    address.SetBase(subnet.str().c_str(), "255.255.255.0");
    interfaces[i] = address.Assign(devices);
  }

  // Setup a TCP packet sink on node 9
  uint16_t port = 8080;
  PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(9));
  sinkApps.Start(Seconds(1.0)); // Start sink at 1.0 seconds
  sinkApps.Stop(Seconds(10.0)); // Stop sink at 10.0 seconds

  // Set up TCP applications on nodes 0 to 8
  for (uint32_t i = 0; i < 9; ++i)
  {
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(nodes.Get(i), TcpSocketFactory::GetTypeId());
    Address sinkAddress(InetSocketAddress(interfaces[i].GetAddress(1), port)); // Each node should target the sink address on its own interface
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket, sinkAddress, 1040, 1, DataRate("1Mbps"), jsonData[i]);
    nodes.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(2.0)); // Start sending at 2.0 seconds
    app->SetStopTime(Seconds(10.0)); // Stop sending at 10.0 seconds

    // Trace packet sent
    app->TraceConnectWithoutContext("Tx", MakeCallback(&SendPacketTrace));
  }

  // Trace packet reception for the sink
  Config::ConnectWithoutContext("/NodeList/9/ApplicationList/*/$ns3::PacketSink/RxWithAddresses",
                                MakeCallback(&ReceivePacketTrace));

   const size_t size = 1024;
    // Allocate a character array to store the directory path
    char buffer[size];

    // Call _getcwd to get the current working directory and store it in buffer
    if (getcwd(buffer, size) != NULL) {
        // print the current working directory
        std::cout << "Current working directory: " << buffer << std::endl;
    }
    else {
        // If _getcwd returns NULL, print an error message
        std::cerr << "Error getting current working directory" << std::endl;
    }
// Enable pcap tracing on all devices -> Directory has to be adapted to the local path
    pointToPoint.EnablePcapAll(pcap::path);

    Simulator::Run();     // Run the simulator
    Simulator::Destroy(); // Clean up after the simulation
    return 0;
}