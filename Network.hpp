#pragma once

/**
 * Implementation is heavily inspired by the boost::asio docs at:
 * https://www.boost.org/doc/libs/1_86_0/doc/html/boost_asio/tutorial/tutdaytime3.html
 */

#include <boost/asio.hpp>

namespace network {

inline std::vector<uint8_t> wait_for_message(uint16_t port) {
  boost::asio::io_context ioc;
  auto acceptor = boost::asio::ip::tcp::acceptor(
      ioc, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
  auto socket = boost::asio::ip::tcp::socket(ioc);

  // bind incoming connection to a port
  acceptor.accept(socket);

  // receive the message length to expect
  std::vector<uint8_t> msg_size(4);
  auto bytes_received = boost::asio::read(
      socket, boost::asio::buffer(msg_size.data(), msg_size.size()));

  // send acknowledgement
  auto bytes_sent = boost::asio::write(socket, boost::asio::buffer("ok"));

  // receive the message itself
  uint32_t *size = std::bit_cast<uint32_t *>(msg_size.data());
  std::vector<uint8_t> msg(*size);
  auto recvd_msg_length =
      boost::asio::read(socket, boost::asio::buffer(msg.data(), msg.size()));

  return msg;
}

inline void send_message(std::string const &ip_address, uint16_t port,
                         std::string const message) {
  namespace ip = boost::asio::ip;
  boost::asio::io_context ioc;
  auto address = ip::address_v4::from_string(ip_address);
  auto socket = ip::tcp::socket(ioc);

  // connect to server
  socket.connect(ip::tcp::endpoint(address, port));

  // send message size
  uint32_t size = message.size();
  socket.send(boost::asio::buffer(&size, sizeof(size)));

  // recv acknowledgement
  std::vector<uint8_t> buffer(2);
  socket.receive(
      boost::asio::buffer(buffer.data(), buffer.size())); // should be "ok"

  // send message
  socket.send(boost::asio::buffer(message));
}

} // namespace network