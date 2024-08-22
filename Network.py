import socketserver
import socket


def wait_for_message(port):
    with socketserver.TCPServer(("127.0.0.1", port), TCPHandler) as server:

        class TCPHandler(socketserver.BaseRequestHandler):
            def handle(self):
                # recv the amount of bytes to expect
                n_bytes = self.request.recv(4)
                n_bytes = int.from_bytes(n_bytes, byteorder="little", signed=False)

                # recv message
                recv_buffer = self.request.recv(n_bytes)
                print("Received {} bytes from {}:".format(self.client_address[0]))

                server.shutdown()

        server.serve_forever()


def send_message(ip, port, message: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        # send length of message
        sock.sendall(len(message).to_bytes(length=4, byteorder="little", signed=False))
        # wait for acknowledge
        sock.recv(2)  # should be "ok"
        # send message
        sock.sendall(message.encode())
