#pragma once

/**
 * Implementation is heavily inspired by the boost::asio docs at:
 * https://www.boost.org/doc/libs/1_86_0/doc/html/boost_asio/tutorial/tutdaytime3.html
 */

#include <boost/asio.hpp>

namespace network {

namespace detail {
class tcp_server {
public:
  tcp_server(boost::asio::io_context &ioc, uint16_t port)
      : m_ioc(ioc), m_acceptor(m_ioc, boost::asio::ip::tcp::endpoint(
                                          boost::asio::ip::tcp::v4(), port)) {
    boost::asio::co_spawn(m_ioc, startAccepting(), [&](std::exception_ptr ep) {
      // executed once, the server is done
      if (ep)
        std::rethrow_exception(ep);
    });
  }

  void run() {
    using namespace std::chrono_literals;

    // don't run out of work
    auto work_guard = boost::asio::make_work_guard(m_ioc);

    // run the server until it is actively stopped
    while (!m_ioc.stopped()) {
      m_ioc.run_for(10ms);
    }
  }

  std::vector<uint8_t> getMessage() const { return m_received_message; }

private:
  boost::asio::awaitable<void> startAccepting() {
    auto error_tuple = boost::asio::as_tuple(boost::asio::use_awaitable);
    auto socket = boost::asio::ip::tcp::socket(m_ioc);
    auto [err] = co_await m_acceptor.async_accept(socket, error_tuple);
    if (err) {
      std::cout << "Cannot accept connection: " << err.what() << std::endl;
      co_return;
    }

    // receive the message length to expect
    std::vector<uint8_t> msg_size(4);
    auto [err2, bytes_received] = co_await boost::asio::async_read(
        socket, boost::asio::buffer(msg_size.data(), msg_size.size()),
        error_tuple);
    if (err2) {
      std::cout << "Cannot receive message size: " << err2.what() << std::endl;
      co_return;
    }

    // send acknowledgement
    auto [err3, bytes_sent] = co_await boost::asio::async_write(
        socket, boost::asio::buffer("ok"), error_tuple);
    if (err3) {
      std::cout << "Cannot send acknowledgement: " << err3.what() << std::endl;
      co_return;
    }

    // receive the message itself
    uint32_t *size = std::bit_cast<uint32_t *>(msg_size.data());
    std::vector<uint8_t> msg(*size);
    auto [err4, recvd_msg_length] = co_await boost::asio::async_read(
        socket, boost::asio::buffer(msg.data(), msg.size()), error_tuple);
    if (err4) {
      std::cout << "Cannot receive message: " << err4.what() << std::endl;
      co_return;
    }

    m_received_message = msg;
    m_ioc.stop();
  }

  boost::asio::io_context &m_ioc;
  boost::asio::ip::tcp::acceptor m_acceptor;
  std::vector<uint8_t> m_received_message;
};
} // namespace detail

inline std::vector<uint8_t> wait_for_message(uint16_t port) {
  boost::asio::io_context ioc;
  auto server = detail::tcp_server(ioc, port);
  server.run();
  return server.getMessage();
}

inline void send_message(std::string const &ip_address, uint16_t port,
                         std::string const message) {
  namespace ip = boost::asio::ip;
  boost::asio::io_context ioc;
  auto address = ip::address_v4::from_string(ip_address);
  auto socket = ip::tcp::socket(ioc, ip::tcp::endpoint(address, port));

  // send message size
  uint32_t size = message.size();
  socket.send(boost::asio::buffer(&size, sizeof(size)));

  // send message
  socket.send(boost::asio::buffer(message));
}

} // namespace network