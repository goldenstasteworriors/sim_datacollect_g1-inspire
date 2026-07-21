#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

constexpr const char* kArmTopic = "rt/arm_sdk";
constexpr const char* kStateTopic = "rt/lowstate";
constexpr int kArmSdkWeightIndex = 29;
constexpr double kControlDt = 0.02;
constexpr double kMinimumDuration = 3.0;
constexpr double kMaximumJointSpeed = 0.35;
constexpr double kMaximumMeasuredSpeed = 0.20;
constexpr double kKp = 60.0;
constexpr double kKd = 1.5;

// Official Unitree Arm SDK order: left arm, right arm, waist.
constexpr std::array<int, 17> kJointIndices = {
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
    12, 13, 14};

// SONICMJ SONIC_G1_DEFAULT_JOINT_POS in the order above.
constexpr std::array<double, 17> kTarget = {
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0};

double SmoothStep(double value) {
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

void FillCommand(
    unitree_hg::msg::dds_::LowCmd_& command,
    const std::array<double, 17>& positions,
    double weight) {
  command.motor_cmd().at(kArmSdkWeightIndex).q(static_cast<float>(weight));
  for (std::size_t i = 0; i < kJointIndices.size(); ++i) {
    auto& motor = command.motor_cmd().at(kJointIndices[i]);
    motor.q(static_cast<float>(positions[i]));
    motor.dq(0.0f);
    motor.kp(static_cast<float>(kKp));
    motor.kd(static_cast<float>(kKd));
    motor.tau(0.0f);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "Usage: g1_initialize_upper_body NETWORK_INTERFACE [--execute]\n";
    return 2;
  }
  const bool execute = argc == 3 && std::string(argv[2]) == "--execute";
  if (argc == 3 && !execute) {
    std::cerr << "The only accepted third argument is --execute\n";
    return 2;
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  std::mutex mutex;
  std::condition_variable condition;
  unitree_hg::msg::dds_::LowState_ state;
  bool received = false;
  auto subscriber = std::make_shared<
      unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>>(kStateTopic);
  subscriber->InitChannel(
      [&](const void* message) {
        std::lock_guard<std::mutex> lock(mutex);
        state = *static_cast<const unitree_hg::msg::dds_::LowState_*>(message);
        received = true;
        condition.notify_one();
      },
      1);
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(lock, std::chrono::seconds(3), [&] { return received; })) {
      std::cerr << "Timed out waiting for rt/lowstate\n";
      return 3;
    }
  }
  if (state.motor_state().size() <= static_cast<std::size_t>(kJointIndices[13])) {
    std::cerr << "LowState does not contain the required 29 body motor slots\n";
    return 4;
  }

  std::array<double, 17> current{};
  double max_delta = 0.0;
  double max_measured_speed = 0.0;
  for (std::size_t i = 0; i < kJointIndices.size(); ++i) {
    const auto& motor = state.motor_state().at(kJointIndices[i]);
    current[i] = motor.q();
    if (!std::isfinite(current[i]) || !std::isfinite(motor.dq())) {
      std::cerr << "LowState contains a non-finite arm/waist value\n";
      return 5;
    }
    max_delta = std::max(max_delta, std::abs(kTarget[i] - current[i]));
    max_measured_speed = std::max(max_measured_speed, std::abs(static_cast<double>(motor.dq())));
  }
  const double duration = std::max(kMinimumDuration, 1.5 * max_delta / kMaximumJointSpeed);
  std::cout << std::fixed << std::setprecision(4)
            << "mode=" << (execute ? "execute" : "dry_run")
            << " mode_machine=" << static_cast<unsigned>(state.mode_machine())
            << " max_measured_speed_rad_s=" << max_measured_speed
            << " max_delta_rad=" << max_delta
            << " duration_s=" << duration << "\n";
  std::cout << "target=[";
  for (std::size_t i = 0; i < kTarget.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << kTarget[i];
  }
  std::cout << "]\n";
  if (!execute) {
    std::cout << "No command publisher was created. Pass --execute only after simulation review.\n";
    return 0;
  }
  if (max_measured_speed > kMaximumMeasuredSpeed) {
    std::cerr << "Robot upper body is still moving; refusing initialization\n";
    return 6;
  }
  std::cout << "Type INITIALIZE and press ENTER to publish rt/arm_sdk: ";
  std::string confirmation;
  std::getline(std::cin, confirmation);
  if (confirmation != "INITIALIZE") {
    std::cerr << "Confirmation mismatch; no publisher was created\n";
    return 7;
  }

  auto publisher = std::make_shared<
      unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>>(kArmTopic);
  publisher->InitChannel();
  unitree_hg::msg::dds_::LowCmd_ command;
  const auto period = std::chrono::milliseconds(20);

  // Acquire Arm SDK weight while holding the measured pose.
  for (int step = 1; step <= 50; ++step) {
    FillCommand(command, current, SmoothStep(step / 50.0));
    publisher->Write(command);
    std::this_thread::sleep_for(period);
  }
  const int motion_steps = std::max(1, static_cast<int>(std::ceil(duration / kControlDt)));
  for (int step = 1; step <= motion_steps; ++step) {
    const double alpha = SmoothStep(static_cast<double>(step) / motion_steps);
    std::array<double, 17> desired{};
    for (std::size_t joint = 0; joint < desired.size(); ++joint) {
      desired[joint] = current[joint] + alpha * (kTarget[joint] - current[joint]);
    }
    FillCommand(command, desired, 1.0);
    publisher->Write(command);
    std::this_thread::sleep_for(period);
  }
  // Hold briefly, then release Arm SDK weight smoothly as in Unitree's example.
  for (int step = 0; step < 100; ++step) {
    FillCommand(command, kTarget, 1.0);
    publisher->Write(command);
    std::this_thread::sleep_for(period);
  }
  for (int step = 49; step >= 0; --step) {
    FillCommand(command, kTarget, SmoothStep(step / 50.0));
    publisher->Write(command);
    std::this_thread::sleep_for(period);
  }
  FillCommand(command, kTarget, 0.0);
  publisher->Write(command);
  std::cout << "Initialization complete and Arm SDK weight released.\n";
  return 0;
}
