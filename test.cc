#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/pcap-file-wrapper.h"

#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace ns3;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("TcpExample");

class MyApp : public Application
{
public:
    void LoadJsonData(std::string path, std::string jsonData[], int size);
    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
               uint32_t nPackets, DataRate dataRate, std::string jsonData);

private:
    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    uint32_t m_nPackets;
    DataRate m_dataRate;
    std::string m_jsonData;
};

void MyApp::LoadJsonData(std::string path, std::string jsonData[], int size)
{
    for (int i = 0; i < size; ++i)
    {
        std::ifstream file(path + "measurement_0_" + std::to_string(i) + ".json");
        if (file)
        {
            std::ostringstream ss;
            ss << file.rdbuf();
            jsonData[i] = ss.str();
        }
        else
        {
            NS_LOG_UNCOND("Error reading file: " + path + "measurement_0_" + std::to_string(i) + ".json");
        }
    }
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, DataRate dataRate, std::string jsonData)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_jsonData = jsonData;

    if (!m_socket)
    {
        NS_LOG_UNCOND("Socket is not created");
        return;
    }

    m_socket->Connect(m_peer);
    NS_LOG_UNCOND("Setup called with JSON Data: " + jsonData);
}

void SendPacketTrace(Ptr<const Packet> packet)
{
    NS_LOG_UNCOND("Packet Sent: " + std::to_string(packet->GetSize()));
}

void ReceivePacketTrace(Ptr<const Packet> packet, const Address& from)
{
    Ipv4Address ipv4Addr = InetSocketAddress::ConvertFrom(from).GetIpv4();
    std::ostringstream oss;
    oss << ipv4Addr;
    NS_LOG_UNCOND("Packet Received from: " + oss.str());
}

int main(int argc, char* argv[])
{
    LogComponentEnable("TcpExample", LOG_LEVEL_INFO);

    std::string jsonData[9];
    MyApp appInstance;
    appInstance.LoadJsonData("../../../../../PycharmProjects/GridComm-Cosimulation/JSON/", jsonData, 9);

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

    pointToPoint.EnablePcapAll("../../../../../PycharmProjects/GridComm-Cosimulation/PCAP/NetSim");

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}