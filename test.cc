#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/network-module.h"
#include "ns3/tap-bridge-module.h"
#include "ns3/internet-module.h"
#include <fstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("TapCsmaVirtualMachineExample");

void SaveTcpPayloadAsJson(Ptr<const Packet> packet, Ptr<NetDevice> device)
{
    // Versuche, die IPv4- und TCP-Header aus dem Paket zu extrahieren
    EthernetHeader ethHeader;
    Ipv4Header ipHeader;
    TcpHeader tcpHeader;
    Ptr<Packet> copy = packet->Copy();

    if (copy->PeekHeader(ethHeader)) {
        if (copy->RemoveHeader(ipHeader)) {
            if (copy->RemoveHeader(tcpHeader)) {
                // Extrahiere die TCP-Nutzlast
                uint32_t dataSize = copy->GetSize();
                uint8_t *buffer = new uint8_t[dataSize];
                copy->CopyData(buffer, dataSize);
                std::string payload(reinterpret_cast<char*>(buffer), dataSize);
                delete[] buffer;

                // Öffne die JSON-Datei im Erstellmodus
                std::ofstream file("messages.json", std::ios::app);
                if (file.is_open())
                {
                    // Erstelle eine JSON-ähnliche Struktur
                    file << "{\n"
                         << "  \"" << ipHeader.GetSource().ToString() << "\": \"" << payload << "\"\n"
                         << "}\n";
                    file.close();
                }
                else
                {
                    NS_LOG_ERROR("Unable to open file for writing JSON.");
                }
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

    // Verbinde die Trace-Callback-Funktion mit dem Netzwerkgerät
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