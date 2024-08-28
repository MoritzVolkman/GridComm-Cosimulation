#pragma once

#include <future>
#include <shared_mutex>
#include <vector>

#include <nlohmann/json.hpp>

#include "Network.hpp"

/**
 * @brief takes a number of json objects and concatenates them into a json array
 * - requires all objects to be of the same type (that is, no mixing of arrays,
 * objects, numbers, etc.)
 *
 * @param jsonData a vector of json objects as raw strings
 * @return nlohmann::json a json array of all objects concatenated
 */
inline auto collect_jsons(std::vector<std::vector<uint8_t>> const &jsonData)
    -> nlohmann::json {
  auto collected_jsons = nlohmann::json{};
  for (auto const &json_str : jsonData) {
    auto json_obj = nlohmann::json::parse(json_str.begin(), json_str.end());
    collected_jsons.push_back(json_obj);
  }

  return collected_jsons;
}

/**
 * @brief opens a couple of ports and waits to receive `n_msgs` messages.
 * Returns after a message on every port has been received
 *
 * @param n_msgs the amount of messages to expect - one on each port
 * @param base_port the base port to listen on for messages. Waiting for 10
 * messages with base port 8000 listens on 8000-8009
 * @return std::vector<std::string> a vector of all received messages sorted by
 * their port
 */
inline auto receive_messages(int n_msgs,
                             uint16_t base_port) -> std::vector<std::string> {
  std::vector<std::string> messages(n_msgs);
  std::vector<std::future<void>> tasks;
  for (int i = 0; i < n_msgs; ++i) {
    auto task = std::async([base_port, i, &messages] {
      auto message = network::wait_for_message(base_port + i);
      messages[i] = std::string{message.begin(), message.end()};
    });
    tasks.push_back(std::move(task));
  }

  // wait for all messages to become available
  for (auto &task : tasks) {
    task.wait();
  }

  return messages;
}