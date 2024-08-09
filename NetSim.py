import json

try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "Error: ns3 Python module not found;"
        " Python bindings may not be enabled"
        " or your PYTHONPATH might not be properly configured"
    )

cmd = ns.CommandLine(__file__)


def simulate_SGMW_TAF14():
    # Set Data Rate and Packet Size for the OnOffApplication
    ns.Config.SetDefault("ns3::OnOffApplication::PacketSize", ns.StringValue("1024"))
    ns.Config.SetDefault("ns3::OnOffApplication::DataRate", ns.StringValue("5Mb/s"))

    # Create the prosumer nodes
    prosumer_nodes = ns.NodeContainer()
    prosumer_nodes.Create(2)

    # Create the grid operator node
    grid_operator_node = ns.NodeContainer()
    grid_operator_node.Create(1)

    point_to_point = ns.PointToPointHelper()
    point_to_point.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
    point_to_point.SetChannelAttribute("Delay", ns.StringValue("2ms"))

    devices = []

    # Create the point-to-point link between the prosumer and the grid operator
    for i in range(0, prosumer_nodes.GetN()):
        device = point_to_point.Install(prosumer_nodes.Get(i), grid_operator_node.Get(0))
        devices.append(device)

    # Install the internet stack on the prosumer and grid operator nodes
    stack = ns.InternetStackHelper()
    stack.Install(prosumer_nodes)
    stack.Install(grid_operator_node)

    address = ns.Ipv4AddressHelper()
    address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))

    interfaces = []
    # Assign IP addresses to the prosumer and grid operator nodes
    for i in range(len(devices)):
        interface = address.Assign(devices[i])
        interfaces.append(interface)

    # Create an address list -> Addresses are saved in order of interfaces
    address_list = []
    for i in range(len(interfaces)):
        address_list.append(interfaces[i].GetAddress(0))
        address_list.append(interfaces[i].GetAddress(1))

    # Create the Server Application on the grid operator node
    port = 8080
    server_helper = ns.PacketSinkHelper("ns3::TcpSocketFactory",
                                        ns.InetSocketAddress(address_list[1], port).ConvertTo())
    server_apps = server_helper.Install(grid_operator_node.Get(0))
    server_apps.Start(ns.Seconds(1.0))
    server_apps.Stop(ns.Seconds(10.0))

    # Create a TCP client on the first prosumer node
    client_helper = ns.BulkSendHelper("ns3::TcpSocketFactory", ns.InetSocketAddress(address_list[0], port).ConvertTo())
    client_helper.SetAttribute("MaxBytes", ns.UintegerValue(1024))
    client_apps = client_helper.Install(prosumer_nodes.Get(0))

    client_apps.Start(ns.Seconds(2.0))
    client_apps.Stop(ns.Seconds(10.0))

    tcp_socket = ns.TcpSocketBase()
    tcp_socket.SetNode(prosumer_nodes.Get(1))
    # The next two throw null pointer exceptions
    # tcp_socket.Bind(ns.InetSocketAddress(address_list[2], port).ConvertTo())
    # tcp_socket.Connect(ns.InetSocketAddress(address_list[3], port).ConvertTo())
    print("TCP Socket has been created")

    # Probably unnecessary code for the packet socket because of TCP socket
    #  -> throws errors at the moment
    # packet_helper = ns.PacketSocketHelper()
    # packet_app = packet_helper.Install(prosumer_nodes.Get(1))
    # packet_socket = packet_app.CreateSocket()
    # packet_socket.Bind(ns.InetSocketAddress(interfaces[1].GetAddress(0), port).ConvertTo())

    # Create a packet -> will be replaced by the measurement data later
    data = "TestPacket"
    data_e = bytearray(data.encode())
    packet = ns.Packet(data_e, 1024)

    tcp_socket.SendTo(packet, 0, ns.InetSocketAddress(address_list[3], port).ConvertTo())
    # tcp_socket.Send(packet, 0)
    tcp_socket.Close()

    # Maybe this is also a valid way to send a packet -> last one is protocol number
    # -> TCP should be 6 according to google, UDP is 17
    devices[1].Get(0).SendFrom(packet, ns.InetSocketAddress(address_list[2], port).ConvertTo(),
                               ns.InetSocketAddress(address_list[3], port).ConvertTo(), 6)

    point_to_point.EnablePcapAll("SGMW_TAF14")

    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    simulate_SGMW_TAF14()




