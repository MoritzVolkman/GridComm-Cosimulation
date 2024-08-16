#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/pcap-file-wrapper.h"

#include <fstream> // Necessary for file operations
#include <sstream>
#include <nlohmann/json.hpp> // Assuming you have access to this library

using namespace ns3;
using json = nlohmann::json; // Define json from nlohmann

// Definiere eine eindeutige Komponente für das Logging
NS_LOG_COMPONENT_DEFINE("TcpExample");

// Custom application to send TCP packets
class MyApp : public Application
{
  // (The rest of your application code remains unchanged...)

  // Invoke my JSON file reading function
  void LoadJsonData(std::string path, std::string jsonData[], int size);

};

// Function to load JSON data from files
void MyApp::LoadJsonData(std::string path, std::string jsonData[], int size)
{
  for (int i = 0; i < size; ++i)
  {
    std::ifstream file(path + "measurement_0_" + std::to_string(i) + ".json");
    if (file)
    {
      std::ostringstream ss;
      ss << file.rdbuf(); // Read file content into string stream
      jsonData[i] = ss.str(); // Store it as a string in array
    }
    else
    {
      NS_LOG_UNCOND("Error reading file: " + path + "measurement_0_" + std::to_string(i) + ".json");
      // Handle error case appropriately (e.g., stop execution or provide default data)
    }
  }
}

int main(int argc, char *argv[])
{
  LogComponentEnable("TcpExample", LOG_LEVEL_INFO);

  // JSON-Strings für die Daten, die versendet werden sollen
  std::string jsonData[9];

  MyApp appInstance;

  // Load JSON data from files
  appInstance.LoadJsonData("../../../PycharmProjects/GridComm-Cosimulation/JSON/", jsonData, 9);

  // The rest of your main function code...
  NodeContainer nodes;
  nodes.Create(10);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
  pointToPoint.SetChannelAttribute("Delay", StringValue("2ms"));

  InternetStackHelper stack;
  stack.Install(nodes);

  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> interfaces(9);

  for (uint32_t i = 0; i < 9; ++i)
  {
    NetDeviceContainer devices = pointToPoint.Install(NodeContainer(nodes.Get(i), nodes.Get(9)));
    std::ostringstream subnet;
    subnet << "10.1." << i + 1 << ".0";
    address.SetBase(subnet.str().c_str(), "255.255.255.0");
    interfaces[i] = address.Assign(devices);
  }

  uint16_t port = 8080;
  PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
  ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(9));
  sinkApps.Start(Seconds(1.0));
  sinkApps.Stop(Seconds(10.0));

  for (uint32_t i = 0; i < 9; ++i)
  {
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(nodes.Get(i), TcpSocketFactory::GetTypeId());
    Address sinkAddress(InetSocketAddress(interfaces[i].GetAddress(1), port));
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket, sinkAddress, 1040, 1, DataRate("1Mbps"), jsonData[i]);
    nodes.Get(i)->AddApplication(app);
    app->SetStartTime(Seconds(2.0));
    app->SetStopTime(Seconds(10.0));

    app->TraceConnectWithoutContext("Tx", MakeCallback(&SendPacketTrace));
  }

  Config::ConnectWithoutContext("/NodeList/9/ApplicationList/*/$ns3::PacketSink/RxWithAddresses",
                                MakeCallback(&ReceivePacketTrace));

  pointToPoint.EnablePcapAll("../../../PycharmProjects/GridComm-Cosimulation/PCAP/NetSim");

  Simulator::Run();
  Simulator::Destroy();
  return 0;
}