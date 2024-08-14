#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/network-module.h"
#include "ns3/tap-bridge-module.h"
#include "ns3/internet-module.h"
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TapCsmaVirtualMachineExample");

void SaveTcpPayloadAsJson(Ptr<const Packet> packet, Ptr<NetDevice> device)
{
    EthernetHeader ethHeader;
    Ipv4Header ipHeader;
    TcpHeader tcpHeader;
    Ptr<Packet> copy = packet->Copy();

    if (copy->PeekHeader(ethHeader)) {
        if (copy->RemoveHeader(ipHeader)) {
            if (copy->RemoveHeader(tcpHeader)) {
                uint32_t dataSize = copy->GetSize();
                uint8_t* buffer = new uint8_t[dataSize];
                copy->CopyData(buffer, dataSize);
                std::string payload(reinterpret_cast<char*>(buffer), dataSize);
                delete[] buffer;

                // Ausgabe des Payloads und der Quell-IP-Adresse in der Konsole
                std::cout << "Source IP: " << ipHeader.GetSource() << ", Payload: " << payload << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);
    cmd.Parse(argc, argv);

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));

    NodeContainer nodes;
    nodes.Create(2);

    CsmaHelper csma;
    NetDeviceContainer devices = csma.Install(nodes);

    TapBridgeHelper tapBridge;
    tapBridge.SetAttribute("Mode", StringValue("UseBridge"));
    tapBridge.SetAttribute("DeviceName", StringValue("tap-left"));
    tapBridge.Install(nodes.Get(0), devices.Get(0));

    devices.Get(0)->GetObject<NetDevice>()->TraceConnectWithoutContext(
        "PromiscRx",
        MakeCallback(&SaveTcpPayloadAsJson));

    tapBridge.SetAttribute("DeviceName", StringValue("tap-right"));
    tapBridge.Install(nodes.Get(1), devices.Get(1));

    Simulator::Stop(Seconds(600.));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}