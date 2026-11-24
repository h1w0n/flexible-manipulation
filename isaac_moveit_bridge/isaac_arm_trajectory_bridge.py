
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState


# OpenMANIPULATOR-X 관절
JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']

# 임시 안전 컷오프: 시작 위치에서 관절당 최대 0.1rad 이동
MAX_MOVE = 0.1
STRICT_GOAL_TOLERANCE = 0.01
SEQUENCE_TEST_GOAL_TOLERANCE = 0.05


class IsaacArmBridge(Node):

    def __init__(self):
        super().__init__('isaac_arm_trajectory_bridge')

        group = ReentrantCallbackGroup()

        self.positions = {}
        self.last_state_time = 0.0
        self.busy = False
        self.sequence_test = os.environ.get('FLEX_SEQUENCE_TEST') == '1'
        self.goal_tolerance = (
            SEQUENCE_TEST_GOAL_TOLERANCE
            if self.sequence_test else STRICT_GOAL_TOLERANCE
        )

        # Isaac Sim으로 관절 명령 발행
        self.pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        # Isaac Sim의 실제 관절 상태 수신
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.state_callback,
            10,
            callback_group=group
        )

        # MoveIt 2의 궤적 실행 요청 수신
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=group
        )

        self.get_logger().info(
            'Isaac Sim 브리지 실행 준비 완료: '
            f"sequence_test={self.sequence_test}, "
            f"goal_tolerance={self.goal_tolerance:.3f}rad"
        )

    def state_callback(self, msg):
        self.positions = dict(zip(msg.name, msg.position))
        self.last_state_time = time.monotonic()

    def publish(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = list(positions)

        self.pub.publish(msg)

    def current(self):
        return [self.positions[j] for j in JOINTS]

    def goal_callback(self, goal):
        points = goal.trajectory.points
        names = goal.trajectory.joint_names

        self.get_logger().warn(
        f'GOAL 진단: joints={list(names)}, '
        f'points={len(points)}, '
        f'busy={self.busy}, '
        f'state_age={time.monotonic() - self.last_state_time:.2f}s, '
        f'duration={points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9 if points else -1:.3f}s'
    )

        # 중복 실행 방지
        if self.busy:
            return GoalResponse.REJECT

        # 관절 이름 검사
        if set(names) != set(JOINTS) or len(names) != 4 or not points:
            return GoalResponse.REJECT

        # 최신 관절 상태 수신 여부 확인
        if time.monotonic() - self.last_state_time > 1.0:
            self.get_logger().error('관절 상태 수신 없음')
            return GoalResponse.REJECT

        try:
            start = [self.positions[j] for j in names]
        except KeyError:
            return GoalResponse.REJECT

        # 궤적 유효성 및 임시 안전 컷오프 검사
        prev_time = -1.0

        for p in points:
            t = (
                p.time_from_start.sec
                + p.time_from_start.nanosec * 1e-9
            )

            if (
                t <= prev_time
                or t > 8.0
                or len(p.positions) != 4
                or not all(math.isfinite(x) for x in p.positions)
                or any(
                    abs(a - b) > MAX_MOVE
                    for a, b in zip(p.positions, start)
                )
            ):
                self.get_logger().warn(
                    '임시 안전 컷오프: 궤적 거부'
                )
                return GoalResponse.REJECT

            prev_time = t

        if prev_time <= 0.0:
            self.get_logger().warn('유효하지 않은 궤적 시간')
            return GoalResponse.REJECT

        self.busy = True
        self.get_logger().info('궤적 요청 수락')

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        result = FollowJointTrajectory.Result()

        names = goal_handle.request.trajectory.joint_names
        points = goal_handle.request.trajectory.points

        try:
            start = [self.positions[j] for j in names]

            begin = time.monotonic()

            total = (
                points[-1].time_from_start.sec
                + points[-1].time_from_start.nanosec * 1e-9
            )

            # 계획된 시간에 따라 관절 명령 전송
            while True:

                # 취소 요청 확인
                if goal_handle.is_cancel_requested:
                    self.publish(self.current())
                    goal_handle.canceled()

                    result.error_code = result.SUCCESSFUL
                    result.error_string = 'Goal canceled'
                    return result

                # 관절 상태 수신 중단 검사
                if time.monotonic() - self.last_state_time > 1.0:
                    goal_handle.abort()

                    result.error_code = (
                        result.PATH_TOLERANCE_VIOLATED
                    )
                    result.error_string = 'Joint state timeout'
                    return result

                elapsed = time.monotonic() - begin

                if elapsed >= total:
                    break

                prev_t = 0.0
                prev = start

                # 시간에 맞춰 중간 관절 위치 보간
                for p in points:
                    t = (
                        p.time_from_start.sec
                        + p.time_from_start.nanosec * 1e-9
                    )

                    if elapsed <= t:
                        ratio = max(
                            0.0,
                            min(
                                1.0,
                                (elapsed - prev_t)
                                / max(t - prev_t, 1e-6)
                            )
                        )

                        target = [
                            a + ratio * (b - a)
                            for a, b in zip(prev, p.positions)
                        ]
                        break

                    prev_t = t
                    prev = p.positions

                else:
                    target = list(points[-1].positions)

                values = dict(zip(names, target))

                self.publish([
                    values[j] for j in JOINTS
                ])

                # ROS2 Humble 비동기 이벤트 루프 오류 수정
                time.sleep(0.04)


            # 최종 목표 관절값
            final = dict(zip(names, points[-1].positions))
            final_positions = [final[j] for j in JOINTS]

            stable_count = 0
            last_counted_state_time = 0.0

            # 최대 2초 동안 목표 위치 유지 및 도달 확인
            for _ in range(40):

                if goal_handle.is_cancel_requested:
                    self.publish(self.current())
                    goal_handle.canceled()

                    result.error_code = result.SUCCESSFUL
                    result.error_string = 'Goal canceled'
                    return result

                # 핵심 변경: 최종 목표를 반복해서 발행
                self.publish(final_positions)

                fresh = (
                    time.monotonic() - self.last_state_time < 1.0
                    and all(j in self.positions for j in JOINTS)
                )

                if not fresh:
                    goal_handle.abort()
                    result.error_code = result.PATH_TOLERANCE_VIOLATED
                    result.error_string = 'Joint state timeout'
                    return result

                reached = (
                    fresh
                    and all(
                        abs(self.positions[j] - final[j]) < self.goal_tolerance
                        for j in JOINTS
                    )
                )

                if reached:
                    if self.last_state_time > last_counted_state_time:
                        stable_count += 1
                        last_counted_state_time = self.last_state_time
                else:
                    stable_count = 0

                # 연속 3회 목표 범위 안에 들어오면 성공
                if stable_count >= 3:
                    goal_handle.succeed()
                    result.error_code = result.SUCCESSFUL

                    self.get_logger().info(
                        '궤적 실행 성공: 목표 관절 위치 확인'
                    )
                    return result

                time.sleep(0.05)


            # 목표 도달 실패
            goal_handle.abort()

            result.error_code = (
                result.GOAL_TOLERANCE_VIOLATED
            )
            result.error_string = 'Goal position not reached'

            self.get_logger().error(
                '목표 위치 도달 확인 실패: '
                + ', '.join(
                    f'{j}={abs(self.positions[j] - final[j]):.4f}rad'
                    for j in JOINTS
                )
            )

            return result

        except Exception as e:
            self.get_logger().error(
                f'궤적 실행 중 예외: {e}'
            )

            goal_handle.abort()

            result.error_code = (
                result.PATH_TOLERANCE_VIOLATED
            )
            result.error_string = str(e)

            return result

        finally:
            self.busy = False


def main():
    rclpy.init()

    node = IsaacArmBridge()

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
