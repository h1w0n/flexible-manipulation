
#!/usr/bin/env python3

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState


LEFT = "gripper_left_joint"
RIGHT = "gripper_right_joint"
JOINTS = [LEFT, RIGHT]

OPEN_POSITION = 0.010
CLOSE_POSITION = -0.010

# 임시 안전 컷오프 및 검증 기준
MIN_POSITION = -0.010
MAX_POSITION = 0.010
POSITION_TOLERANCE = 0.002
ACTION_TIMEOUT = 7.0


class IsaacGripperBridge(Node):

    def __init__(self):
        super().__init__("isaac_gripper_bridge")

        self.group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.busy = False

        self.positions = {}
        self.efforts = {}
        self.last_state_time = 0.0

        self.publisher = self.create_publisher(
            JointState,
            "/joint_commands",
            10,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.state_callback,
            qos_profile_sensor_data,
            callback_group=self.group,
        )

        self.server = ActionServer(
            self,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.group,
        )

        self.get_logger().info(
            "Gripper Action Server READY"
        )

    def state_callback(self, msg):
        if not all(j in msg.name for j in JOINTS):
            return

        self.positions = dict(zip(msg.name, msg.position))
        self.efforts = dict(zip(msg.name, msg.effort))
        self.last_state_time = time.monotonic()

    def publish_position(self, left, right):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = [left, right]
        self.publisher.publish(msg)

    def goal_callback(self, request):
        position = request.command.position

        if not math.isfinite(position):
            return GoalResponse.REJECT

        if not MIN_POSITION <= position <= MAX_POSITION:
            self.get_logger().warn("임시 안전 컷오프: 범위 초과")
            return GoalResponse.REJECT

        if time.monotonic() - self.last_state_time > 1.0:
            self.get_logger().warn("관절 상태 수신 없음")
            return GoalResponse.REJECT

        with self.lock:
            if self.busy:
                return GoalResponse.REJECT
            self.busy = True

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def make_result(self, reached):
        result = GripperCommand.Result()
        result.position = self.positions.get(LEFT, 0.0)
        result.effort = self.efforts.get(LEFT, 0.0)
        result.stalled = False
        result.reached_goal = reached
        return result

    def execute_callback(self, goal_handle):
        target = goal_handle.request.command.position

        self.get_logger().info(
            f"Gripper target = {target:.4f} m"
        )

        if goal_handle.request.command.max_effort > 0:
            self.get_logger().warn(
                "max_effort 제한은 아직 구현되지 않았습니다."
            )

        started = time.monotonic()
        stable_count = 0

        try:
            while time.monotonic() - started < ACTION_TIMEOUT:

                if goal_handle.is_cancel_requested:
                    left = self.positions.get(LEFT, 0.0)
                    right = self.positions.get(RIGHT, 0.0)
                    self.publish_position(left, right)
                    goal_handle.canceled()
                    return self.make_result(False)

                # 양쪽 관절에 동일한 목표값 전달
                self.publish_position(target, target)

                fresh = (
                    time.monotonic() - self.last_state_time < 1.0
                )

                reached = fresh and all(
                    abs(self.positions.get(j, 999) - target)
                    <= POSITION_TOLERANCE
                    for j in JOINTS
                )

                stable_count = (
                    stable_count + 1 if reached else 0
                )

                if stable_count >= 3:
                    goal_handle.succeed()
                    self.get_logger().info(
                        "Gripper target reached"
                    )
                    return self.make_result(True)

                feedback = GripperCommand.Feedback()
                feedback.position = self.positions.get(
                    LEFT, 0.0
                )
                feedback.effort = self.efforts.get(
                    LEFT, 0.0
                )
                feedback.stalled = False
                feedback.reached_goal = False
                goal_handle.publish_feedback(feedback)

                time.sleep(0.05)

            self.get_logger().warn(
                "목표 미도달: "
                f"left={self.positions.get(LEFT)}, "
                f"right={self.positions.get(RIGHT)}"
            )

            # 시간 초과 시 현재 관절 위치 유지 시도
            self.publish_position(
                self.positions.get(LEFT, 0.0),
                self.positions.get(RIGHT, 0.0),
            )
            goal_handle.abort()
            return self.make_result(False)

        finally:
            with self.lock:
                self.busy = False


def main():
    rclpy.init()

    node = IsaacGripperBridge()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
