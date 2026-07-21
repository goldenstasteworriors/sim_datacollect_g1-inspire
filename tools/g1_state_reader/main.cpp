#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

constexpr const char* kTopic = "rt/lowstate";
constexpr const char* kPrefix = "G1_LOWSTATE_JSON ";
constexpr std::array<int, 7> kRightArmIndices = {22, 23, 24, 25, 26, 27, 28};
constexpr std::array<int, 3> kWaistIndices = {12, 13, 14};
constexpr std::array<int, 29> kBodyIndices = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

template <typename MotorContainer, std::size_t N>
void PrintMotorField(
    const MotorContainer& motors,
    const std::array<int, N>& indices,
    bool velocity) {
  std::cout << "[";
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (i != 0) std::cout << ",";
    const auto& motor = motors.at(indices[i]);
    std::cout << (velocity ? motor.dq() : motor.q());
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: g1_read_lowstate NETWORK_INTERFACE [TIMEOUT_SECONDS]\n";
    return 2;
  }
  const std::string network_interface = argv[1];
  const double timeout_seconds = argc == 3 ? std::atof(argv[2]) : 3.0;
  if (!(timeout_seconds > 0.0 && timeout_seconds <= 30.0)) {
    std::cerr << "TIMEOUT_SECONDS must be in (0, 30]\n";
    return 2;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
  std::mutex mutex;
  std::condition_variable condition;
  unitree_hg::msg::dds_::LowState_ snapshot;
  bool received = false;

  auto subscriber = std::make_shared<
      unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(kTopic);
  subscriber->InitChannel(
      [&](const void* message) {
        std::lock_guard<std::mutex> lock(mutex);
        snapshot = *static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
        received = true;
        condition.notify_one();
      },
      1);

  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(
            lock, std::chrono::duration<double>(timeout_seconds), [&] { return received; })) {
      std::cerr << "Timed out waiting for " << kTopic << " on " << network_interface << "\n";
      return 3;
    }
  }

  const auto& motors = snapshot.motor_state();
  if (motors.size() <= static_cast<std::size_t>(kRightArmIndices.back())) {
    std::cerr << "LowState contains only " << motors.size() << " motor states\n";
    return 4;
  }
  const double timestamp = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  std::cout << std::setprecision(10) << kPrefix << "{\"timestamp\":" << timestamp
            << ",\"topic\":\"" << kTopic << "\",\"network_interface\":\""
            << network_interface << "\",\"mode_machine\":"
            << static_cast<unsigned>(snapshot.mode_machine()) << ",\"right_arm_q\":";
  PrintMotorField(motors, kRightArmIndices, false);
  std::cout << ",\"right_arm_dq\":";
  PrintMotorField(motors, kRightArmIndices, true);
  std::cout << ",\"waist_q\":";
  PrintMotorField(motors, kWaistIndices, false);
  std::cout << ",\"waist_dq\":";
  PrintMotorField(motors, kWaistIndices, true);
  std::cout << ",\"body_q\":";
  PrintMotorField(motors, kBodyIndices, false);
  std::cout << ",\"body_dq\":";
  PrintMotorField(motors, kBodyIndices, true);
  std::cout << "}" << std::endl;
  return 0;
}
