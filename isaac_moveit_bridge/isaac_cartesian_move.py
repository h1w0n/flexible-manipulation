
#!/usr/bin/env python3

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from action_msgs.msg import GoalStatus


JOINTS = ["joint1", "joint2", "joint3", "joint4"]
BASE_FRAME = "base_footprint"
TIP_LINK = "end_effector_link"

# 임시 안전 컷오프
MAX_CARTESIAN_MOVE = 0.02   # 한 번에 최대 2cm
MAX_JOINT_MOVE = 0.06       # 관절당 최대 0.06rad


def wait_future(node, future, label, timeout=10.0):
    rclpy.spin_until_future_complete(
        node, future, timeout_sec=timeout
    )
    if not future.done():
        raise RuntimeError(f"{label}: 응답 시간 초과")

    result = future.result()
    if result is None:
        raise RuntimeError(f"{label}: 응답 없음")

    return result


class CartesianMove(Node):

    def __init__(self):
        super().__init__("isaac_cartesian_move")

        self.joint_state = None
        self.state_received_at = 0.0

        self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_state,
            qos_profile_sensor_data,
        )

        self.fk_client = self.create_client(
            GetPositionFK, "/compute_fk"
        )

        self.ik_client = self.create_client(
            GetPositionIK, "/compute_ik"
        )

        self.move_client = ActionClient(
            self, MoveGroup, "/move_action"
        )

    def on_joint_state(self, msg):
        if all(j in msg.name for j in JOINTS):
            self.joint_state = msg
            self.state_received_at = time.monotonic()

    def get_current_state(self):
        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)

            if (
                self.joint_state is not None
                and time.monotonic()
                - self.state_received_at < 1.0
            ):
                msg = JointState()
                msg.name = list(self.joint_state.name)
                msg.position = list(self.joint_state.position)
                return msg

        raise RuntimeError("/joint_states 수신 실패")

    def get_fk(self, state):
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_FRAME
        request.fk_link_names = [TIP_LINK]
        request.robot_state.joint_state = state

        response = wait_future(
            self,
            self.fk_client.call_async(request),
            "FK 계산",
        )

        if response.error_code.val != 1:
            raise RuntimeError(
                f"FK 실패: {response.error_code.val}"
            )

        return response.pose_stamped[0].pose

    def solve_ik(self, state, target, orientation):
        request = GetPositionIK.Request()
        ik = request.ik_request

        ik.group_name = "arm"
        ik.robot_state.joint_state = state
        ik.ik_link_name = TIP_LINK

        ik.pose_stamped.header.frame_id = BASE_FRAME
        ik.pose_stamped.pose.position.x = target[0]
        ik.pose_stamped.pose.position.y = target[1]
        ik.pose_stamped.pose.position.z = target[2]
        ik.pose_stamped.pose.orientation = orientation

        ik.timeout.sec = 1
        ik.avoid_collisions = False

        response = wait_future(
            self,
            self.ik_client.call_async(request),
            "IK 계산",
        )

        if response.error_code.val != 1:
            raise RuntimeError(
                f"IK 실패: {response.error_code.val}"
            )

        solution = response.solution.joint_state
        return dict(zip(solution.name, solution.position))

    def execute(self, target_joints):
        goal = MoveGroup.Goal()
        request = goal.request

        request.group_name = "arm"
        request.start_state.is_diff = True

        constraints = Constraints()

        for name in JOINTS:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target_joints[name]
            jc.tolerance_above = 0.002
            jc.tolerance_below = 0.002
            jc.weight = 1.0

            constraints.joint_constraints.append(jc)

        request.goal_constraints = [constraints]
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.05
        request.max_acceleration_scaling_factor = 0.05

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        print("\n[4] MoveIt 경로 계획 및 실행 요청")

        handle = wait_future(
            self,
            self.move_client.send_goal_async(goal),
            "MoveIt 목표 접수",
        )

        if not handle.accepted:
            raise RuntimeError("MoveIt 목표 요청 거부")

        result_future = handle.get_result_async()

        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=35.0
        )

        if not result_future.done():
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=3.0
            )
            raise RuntimeError(
                "실행 시간 초과: 취소 요청 전송"
            )

        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("MoveIt 결과 없음")

        code = wrapped.result.error_code.val

        print(f"MoveIt 결과 코드: {code}")
        print(f"Action 상태: {wrapped.status}")

        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or code != 1
        ):
            raise RuntimeError(
                "MoveIt 경로 계획 또는 실행 실패"
            )

    def run(self, args):

        for client, label in [
            (self.fk_client, "FK"),
            (self.ik_client, "IK"),
        ]:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{label} 서비스 없음")

        if not self.move_client.wait_for_server(
            timeout_sec=5.0
        ):
            raise RuntimeError("MoveIt Action 서버 없음")

        state = self.get_current_state()
        current_joints = dict(
            zip(state.name, state.position)
        )

        pose = self.get_fk(state)
        current_xyz = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]

        print("\n[1] 현재 말단 위치")
        print(
            f"x={current_xyz[0]:.5f}, "
            f"y={current_xyz[1]:.5f}, "
            f"z={current_xyz[2]:.5f}"
        )

        # 절대 좌표 또는 현재 위치 기준 상대 이동
        if args.x is not None:
            target = [args.x, args.y, args.z]
        else:
            target = [
                current_xyz[0] + args.dx,
                current_xyz[1] + args.dy,
                current_xyz[2] + args.dz,
            ]

        distance = math.dist(current_xyz, target)

        print("\n[2] 목표 말단 위치")
        print(
            f"x={target[0]:.5f}, "
            f"y={target[1]:.5f}, "
            f"z={target[2]:.5f}"
        )

        # 임시 안전 컷오프: 목표 위치 이동 거리
        if distance > MAX_CARTESIAN_MOVE:
            raise RuntimeError(
                "임시 안전 컷오프: 목표 이동 거리 2cm 초과"
            )

        solution = self.solve_ik(
            state, target, pose.orientation
        )

        print("\n[3] IK 관절값 및 변화량")

        for name in JOINTS:
            delta = solution[name] - current_joints[name]

            print(
                f"{name}: {solution[name]:.5f} rad "
                f"(변화량 {delta:+.5f})"
            )

            if abs(delta) > MAX_JOINT_MOVE:
                raise RuntimeError(
                    f"임시 안전 컷오프: {name} 변화량 초과"
                )

        # 실제 이동 전 최신 상태 재확인
        latest = self.get_current_state()
        latest_joints = dict(
            zip(latest.name, latest.position)
        )

        for name in JOINTS:
            if abs(
                latest_joints[name] - current_joints[name]
            ) > 0.02:
                raise RuntimeError(
                    "로봇 자세가 변경됨. 다시 실행 필요"
                )

        self.execute(solution)

        # 실행 후 새 관절 상태 수신
        previous_time = self.state_received_at
        deadline = time.monotonic() + 3.0

        while (
            time.monotonic() < deadline
            and self.state_received_at <= previous_time
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        final_state = self.get_current_state()
        final_pose = self.get_fk(final_state)

        final_xyz = [
            final_pose.position.x,
            final_pose.position.y,
            final_pose.position.z,
        ]

        error_mm = math.dist(final_xyz, target) * 1000

        print("\n[5] 최종 FK 검증")
        print(
            f"x={final_xyz[0]:.5f}, "
            f"y={final_xyz[1]:.5f}, "
            f"z={final_xyz[2]:.5f}"
        )
        print(f"목표 위치 오차: {error_mm:.2f} mm")

        if error_mm > 5:
            print("주의: 목표와 5mm 이상 차이가 있습니다.")

        print("\n자동 이동 절차 완료!")


def main():
    parser = argparse.ArgumentParser()

    # 절대 목표 좌표
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)

    # 현재 위치 기준 상대 이동량
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)

    args = parser.parse_args()

    values = [args.x, args.y, args.z]
    if any(v is not None for v in values):
        if not all(v is not None for v in values):
            parser.error("--x, --y, --z는 모두 입력해야 합니다.")
        if any(
            v != 0.0 for v in (args.dx, args.dy, args.dz)
        ):
            parser.error("절대 좌표와 상대 이동은 혼용할 수 없습니다.")

    rclpy.init()
    node = CartesianMove()

    try:
        node.run(args)
    except Exception as e:
        print(f"\n[중단] {e}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
