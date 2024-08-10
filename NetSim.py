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

"""
Structure of the Simulated network should look somewhat like this:


    Prosumer 1  <--Point-to-Point--> Bridge 1 <--CSMA--> Bridge 2 <--Point-to-Point--> Grid Operator
    Prosumer 2  <--Point-to-Point--> Bridge 1 <--CSMA--> Bridge 2 <--Point-to-Point--> Grid Operator
    
    
    TODO: Müsste aber vielleicht eher so Aussehen:
    
    Prosumer 1  <--CSMA--> 
                            Bridge 1 <--Point-to-Point--> Bridge 2 <--Point-to-Point--> Grid Operator
    Prosumer 2  <--CSMA--> 
    
etc.
Basically, there is the bridge connection, which on the one side is connected to all the prosumers 
and on the other side to the grid operator.
    
"""


def simulate_SGMW_TAF14():
    # Set Data Rate and Packet Size for the OnOffApplication
    ns.Config.SetDefault("ns3::OnOffApplication::PacketSize", ns.StringValue("1024"))
    ns.Config.SetDefault("ns3::OnOffApplication::DataRate", ns.StringValue("5Mb/s"))

    # Create the prosumer nodes
    prosumer_nodes = ns.NodeContainer()
    prosumer_nodes.Create(2)

    # Create some bridge nodes
    bridge_nodes = ns.NodeContainer()
    bridge_nodes.Create(2)

    # Create the CSMA devices for the bridge nodes
    csma = ns.CsmaHelper()
    csma.SetChannelAttribute("DataRate", ns.StringValue("100Mbps"))
    csma.SetChannelAttribute("Delay", ns.StringValue("2ms"))
    bridge_devices = csma.Install(bridge_nodes)

    # Create the grid operator node
    grid_operator_node = ns.NodeContainer()
    grid_operator_node.Create(1)

    # Create the point-to-point link between the prosumer and the bridge nodes and the bridge and grid operator
    point_to_point = ns.PointToPointHelper()
    point_to_point.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
    point_to_point.SetChannelAttribute("Delay", ns.StringValue("2ms"))

    prosumer_devices = []

    # Create the point-to-point link between the prosumer and the first bridge node
    for i in range(0, prosumer_nodes.GetN()):
        device = point_to_point.Install(prosumer_nodes.Get(i), bridge_nodes.Get(0))
        prosumer_devices.append(device)

    # Create the point-to-point link between the bridge nodes and the grid operator
    grid_operator_device = point_to_point.Install(bridge_nodes.Get(1), grid_operator_node.Get(0))

    tap_bridge = ns.TapBridgeHelper()
    # tap_bridge.SetAttribute("Mode", ns.StringValue("ConfigureLocal"))
    # tap_bridge.Install(grid_operator_node.Get(0), "ns3")

    # Install the internet stack on the prosumer and grid operator nodes
    stack = ns.InternetStackHelper()
    olsr = ns.OlsrHelper()
    # Somehow this stops the simulation from continuing
    stack.SetRoutingHelper(olsr)
    stack.InstallAll()

    # Declare the IP address space
    address = ns.Ipv4AddressHelper()
    address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))

    # Assign IP addresses to the bridge nodes and the grid operator node
    bridge_addresses = address.Assign(bridge_devices)
    grid_operator_address = address.Assign(grid_operator_device)

    interfaces = []
    # Assign IP addresses to the prosumer and bridge nodes
    for i in range(len(prosumer_devices)):
        interface = address.Assign(prosumer_devices[i])
        interfaces.append(interface)

    # Create an address list -> Addresses are saved in order of interfaces
    address_list = []
    for i in range(len(interfaces)):
        address_list.append(interfaces[i].GetAddress(0))
        address_list.append(interfaces[i].GetAddress(1))

    # Create the Server Application on the grid operator node
    port = 8080
    server_helper = ns.PacketSinkHelper("ns3::TcpSocketFactory",
                                        ns.InetSocketAddress(grid_operator_address.GetAddress(1), port).ConvertTo())
    server_apps = server_helper.Install(grid_operator_node.Get(0))
    grid_operator_node.Get(0).AddApplication(server_apps.Get(0))
    point_to_point.EnablePcap("GO", grid_operator_node.Get(0), True)
    server_apps.Start(ns.Seconds(1.0))
    server_apps.Stop(ns.Seconds(10.0))

    # Create a TCP client on the first prosumer node
    client_helper = ns.OnOffHelper("ns3::TcpSocketFactory", ns.InetSocketAddress(address_list[0], port).ConvertTo())
    client_helper.SetAttribute("MaxBytes", ns.UintegerValue(1024))
    client_apps = client_helper.Install(prosumer_nodes.Get(0))
    prosumer_nodes.Get(0).AddApplication(client_apps.Get(0))
    point_to_point.EnablePcap("Pro", prosumer_nodes.Get(0), True)

    client_apps.Start(ns.Seconds(2.0))
    client_apps.Stop(ns.Seconds(10.0))

    # tcp_socket = ns.TcpSocketBase()
    # tcp_socket.SetNode(prosumer_nodes.Get(1))

    # The next two throw null pointer exceptions
    # tcp_socket.Bind(ns.InetSocketAddress(address_list[2], port).ConvertTo())
    # tcp_socket.Connect(ns.InetSocketAddress(address_list[3], port).ConvertTo())

    """ not really working (or necessary?) at the moment
    # Probably unnecessary code for the packet socket because of TCP socket
    #  -> throws errors at the moment
    # packet_helper = ns.PacketSocketHelper()
    # packet_app = packet_helper.Install(prosumer_nodes.Get(1))
    # packet_socket = packet_app.CreateSocket()
    # packet_socket.Bind(ns.InetSocketAddress(interfaces[1].GetAddress(0), port).ConvertTo())
    """

    # Create a packet -> will be replaced by the measurement data later
    data = "TestPacket"
    data_e = bytearray(data.encode())
    packet = ns.Packet(data_e, 1024)

    # tcp_socket.SendTo(packet, 0, ns.InetSocketAddress(address_list[3], port).ConvertTo())
    # tcp_socket.Send(packet, 0)
    # tcp_socket.Close()

    # Maybe this is also a valid way to send a packet -> last one is protocol number
    # -> TCP should be 6 according to google, UDP is 17
    # print(prosumer_devices[1].Get(0).GetAddress())
    # prosumer_devices[1].Get(0).SendFrom(packet, ns.InetSocketAddress(address_list[2], port).ConvertTo(),
    #                          ns.InetSocketAddress(grid_operator_address.GetAddress(1), port).ConvertTo(), 6)
    # print(devices[1].Get(1).Receive(packet))
    # Still nothing is received
    # print(server_apps.Get(0).GetTotalRx())

    # point_to_point.EnablePcapAll("SGMW_TAF14")

    ns.Simulator.Stop(ns.Seconds(11.0))
    ns.Simulator.Run()
    ns.Simulator.Destroy()

    print("Simulation has ended")
if __name__ == "__main__":
    simulate_SGMW_TAF14()




