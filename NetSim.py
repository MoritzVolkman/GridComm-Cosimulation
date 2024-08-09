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

    # Assign IP addresses to the prosumer and grid operator nodes
    for i in range(len(devices)):
        interfaces = address.Assign(devices[i])

    # Create the Server Application on the grid operator node
    port = 8080
    print(type(ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo()))
    server_helper = ns.PacketSinkHelper("ns3::TcpSocketFactory", ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo())
    server_apps = server_helper.Install(grid_operator_node.Get(0))
    server_apps.Start(ns.Seconds(1.0))
    server_apps.Stop(ns.Seconds(10.0))

    # Create a TCP client on the first prosumer node
    client_helper = ns.BulkSendHelper("ns3::TcpSocketFactory", ns.InetSocketAddress(interfaces.GetAddress(1), port).ConvertTo())
    client_helper.SetAttribute("MaxBytes", ns.UintegerValue(1024))

    client_apps = client_helper.Install(prosumer_nodes.Get(0))
    client_apps.Start(ns.Seconds(2.0))
    client_apps.Stop(ns.Seconds(10.0))

    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    simulate_SGMW_TAF14()




