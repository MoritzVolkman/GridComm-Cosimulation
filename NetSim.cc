#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/pcap-file-wrapper.h"

#include <nlohmann/json.hpp>

#include "NetSimApp.hpp"
#include "NetSimSinkApp.hpp"
#include "helpers.hpp"

#include <fstream>
#include <sstream>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("TcpExample");

void SendPacketTrace(Ptr<const Packet> packet)
{
    NS_LOG_UNCOND("Packet Sent: " + std::to_string(packet->GetSize()));
}

void ReceivePacketTrace(Ptr<const Packet> packet, const Address& from, const Address& to)
{
    Ipv4Address ipv4AddrFrom = InetSocketAddress::ConvertFrom(from).GetIpv4();
    Ipv4Address ipv4AddrTo = InetSocketAddress::ConvertFrom(to).GetIpv4();
    std::ostringstream oss;
    oss << "Packet Received from: " << ipv4AddrFrom << " to: " << ipv4AddrTo;
    NS_LOG_UNCOND(oss.str());

    // Extract the payload from the packet and convert it to a string
    std::vector<uint8_t> buffer(packet->GetSize());
    packet->CopyData(buffer.data(), buffer.size());
    std::string jsonData(buffer.begin(), buffer.end());

    // Parse the string as JSON and save it to a file
    try {
        json receivedJson = json::parse(jsonData);

        // Open "grid_data.json" in append mode to add the received JSON data
        std::ofstream outFile("JSON/grid_data.json", std::ios::app);

        // Write the JSON data to the file
        outFile << receivedJson.dump() << std::endl;
    } catch (json::parse_error& e) {
        NS_LOG_UNCOND("Failed to parse JSON: " + std::string(e.what()));
    }
}

void simulation_loop(uint32_t n_nodes) {

  // Receive `n_nodes` amount of messages from the GridSim.py for further
  // processing
  auto measurements = receive_messages(n_nodes, 8081);

  NetSimApp appInstance;

  NodeContainer nodes;
  nodes.Create(n_nodes +
               1); // create a node for every measurement + one for the operator

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
  pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

  InternetStackHelper stack;
  stack.Install(nodes);

  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> interfaces(n_nodes);

  // connect all nodes to the operator node
  for (uint32_t i = 0; i < n_nodes; ++i) {
    NetDeviceContainer devices =
        pointToPoint.Install(NodeContainer(nodes.Get(i), nodes.Get(n_nodes)));
    address.SetBase(std::format("10.0.{}.0", i + 1).c_str(), "255.255.255.0");
    interfaces[i] = address.Assign(devices);
  }

  // install a packet sink on the operator node for every sender
  uint16_t port = 8080;
  std::vector<Ptr<Socket>> sockets;
  for (int i = 0; i < n_nodes; ++i) {
    Ptr<Socket> sinkSocket =
        Socket::CreateSocket(nodes.Get(n_nodes), TcpSocketFactory::GetTypeId());
    sinkSocket->Bind(InetSocketAddress(interfaces[i].GetAddress(1), 8080));
    sockets.push_back(sinkSocket);
  }
  Ptr<SinkApp> sinkApp = Create<SinkApp>(sockets);
  nodes.Get(n_nodes)->AddApplication(sinkApp);
  sinkApp->SetStartTime(Seconds(0.5));
  sinkApp->SetStopTime(Seconds(20.));

  PacketSinkHelper packetSinkHelper(
      "ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(n_nodes));
  sinkApps.Start(Seconds(1.0));
  sinkApps.Stop(Seconds(10.0));

  for (uint32_t i = 0; i < n_nodes; ++i) {
    Ptr<Socket> ns3TcpSocket =
        Socket::CreateSocket(nodes.Get(i), TcpSocketFactory::GetTypeId());
    Address sinkAddress(InetSocketAddress(interfaces[i].GetAddress(1), port));
    Ptr<NetSimApp> app = CreateObject<NetSimApp>();
    app->Setup(ns3TcpSocket, sinkAddress, 1040, DataRate("1Mbps"),
               measurements[i]);
    nodes.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(2.0));
    app->SetStopTime(Seconds(10.0));

    app->TraceConnectWithoutContext("Tx", MakeCallback(&SendPacketTrace));
  }

  nodes.Get(n_nodes)->TraceConnectWithoutContext(
      "RxWithAddresses", MakeCallback(&ReceivePacketTrace));

  //   Config::ConnectWithoutContext(
  //       std::format(
  //           "/NodeList/{}/ApplicationList/*/$ns3::PacketSink/RxWithAddresses",
  //           n_nodes)
  //           .c_str(),
  //       MakeCallback(&ReceivePacketTrace));

  pointToPoint.EnablePcapAll("PCAP/NetSim");

  Simulator::Run();
  Simulator::Destroy();

  auto recvd_measurements = sinkApp->getMeasurements();
  auto aggregated_measurements = collect_jsons(recvd_measurements);

  // Send the JSON data to the Python simulation via netcat
  network::send_message("127.0.0.1", 10000, aggregated_measurements.dump());
}

int main(int argc, char* argv[])
{
  // receive the number of simulation nodes nodes
  auto n_nodes_str = network::wait_for_message(10000);
  auto n_nodes = std::stoi(std::string{n_nodes_str.begin(), n_nodes_str.end()});

  // keep reading in measurements, running the netsim and responding with the
  // aggregated results
  LogComponentEnable("TcpExample", LOG_LEVEL_INFO);
  while (true) {
    simulation_loop(n_nodes);
  }
    return 0;
}