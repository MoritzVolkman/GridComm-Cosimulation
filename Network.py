import socketserver
import socket
import threading

def wait_for_message(port):
    on_complete = None
    msg = None

    class TCPHandler(socketserver.BaseRequestHandler):
        def handle(self):
            nonlocal msg
            # recv the amount of bytes to expect
            n_bytes = self.request.recv(4)
            n_bytes = int.from_bytes(n_bytes, byteorder="little", signed=False)

            # send acknowledgement
            self.request.send("ok".encode())

            # recv message
            msg = self.request.recv(n_bytes)
            print(f"Received {n_bytes} bytes from {self.client_address[0]}")

            on_complete()

    with socketserver.TCPServer(("127.0.0.1", port), TCPHandler) as server:

        on_complete = lambda: threading.Thread(target=lambda: server.shutdown()).start()

        server.serve_forever()
        return msg


def send_message(ip, port, message: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        # send length of message
        sock.sendall(len(message).to_bytes(length=4, byteorder="little", signed=False))
        # wait for acknowledge
        sock.recv(2)  # should be "ok"
        # send message
        sock.sendall(message.encode())
