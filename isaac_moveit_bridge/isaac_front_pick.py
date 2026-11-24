
#!/usr/bin/env python3

import argparse
import math

import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand

# 기존에 성공한 좌표 이동 프로그램 재사용
from isaac_cartesian_move import CartesianMove

GRIPPER_ACTION = "/gripper_controller/gripper_cmd"

OPEN_POSITION = 0.010
CLOSE_POSITION = -0.010

# 임시 안전 컷오프
MAX_STEP = 0.020        # 이동 1회당 최대 2cm
MAX_ERROR_MM = 10.0      # Pick 전 말단 위치 오차
SEQUENCE_TEST_MAX_ERROR_MM = 15.0  # 빈 공간 통합 시험 전용 상한


def get_xyz(node, diagnose_fk=False):
    state = node.get_current_state()
    pose = node.get_fk(state, diagnose=diagnose_fk)

    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ]


def send_gripper(node, client, position, label):
    print(f"\n[GRIPPER] {label}")

    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = 0.0

    future = client.send_goal_async(goal)

    rclpy.spin_until_future_complete(
        node, future, timeout_sec=5.0
    )

    if not future.done():
        raise RuntimeError("그리퍼 목표 접수 시간 초과")

    handle = future.result()

    if handle is None or not handle.accepted:
        raise RuntimeError("그리퍼 명령 거부")

    result_future = handle.get_result_async()

    rclpy.spin_until_future_complete(
        node, result_future, timeout_sec=12.0
    )

    if not result_future.done():
        cancel_future = handle.cancel_goal_async()
        rclpy.spin_until_future_complete(
            node, cancel_future, timeout_sec=3.0
        )
        raise RuntimeError("그리퍼 실행 시간 초과")

    wrapped = result_future.result()

    if wrapped is None:
        raise RuntimeError("그리퍼 결과 수신 실패")

    if (
        wrapped.status != GoalStatus.STATUS_SUCCEEDED
        or not wrapped.result.reached_goal
    ):
        raise RuntimeError(
            f"그리퍼 {label} 실패: "
            f"status={wrapped.status}"
        )

    print(f"[OK] {label} 완료")


def move_xyz(node, target, label, sequence_test=False):
    print(f"\n[ARM] {label}")
    print(
        f"목표: ({target[0]:.4f}, "
        f"{target[1]:.4f}, {target[2]:.4f})"
    )

    current = get_xyz(node)
    distance = math.dist(current, target)

    if distance > MAX_STEP:
        raise RuntimeError(
            f"임시 안전 컷오프: "
            f"이동 거리 {distance*1000:.1f}mm > 20mm"
        )

    # 기존 CartesianMove.run() 재사용
    args = argparse.Namespace(
        x=target[0],
        y=target[1],
        z=target[2],
        dx=0.0,
        dy=0.0,
        dz=0.0,
    )

    result = node.run(args)
    actual = result["actual_xyz"]
    error_mm = result["error_mm"]

    print(f"[검증] 실제 좌표: {tuple(round(v, 5) for v in actual)}")
    print(f"[검증] 위치 오차: {error_mm:.2f}mm")
    print(
        f"[검증] MoveIt: status={result['action_status']}, "
        f"code={result['moveit_code']}"
    )

    limit_mm = (
        SEQUENCE_TEST_MAX_ERROR_MM if sequence_test else MAX_ERROR_MM
    )
    if error_mm > limit_mm:
        raise RuntimeError(
            f"목표 위치 오차 {error_mm:.2f}mm가 "
            f"허용 상한 {limit_mm:.1f}mm를 초과했습니다. "
            "다음 동작을 중단합니다."
        )

    if sequence_test and error_mm > MAX_ERROR_MM:
        print(
            f"[경고] 정밀 Pick 기준 {MAX_ERROR_MM:.1f}mm 초과. "
            "빈 공간 시퀀스 검증만 계속합니다."
        )

    print(f"[OK] {label} 완료")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)

    parser.add_argument(
        "--approach", type=float, default=0.010
    )
    parser.add_argument(
        "--lift", type=float, default=0.010
    )
    parser.add_argument(
        "--dry-run", action="store_true"
    )
    parser.add_argument(
        "--sequence-test", action="store_true",
        help="빈 공간 통합 시험: 10~15mm 오차는 경고, 15mm 초과는 중단",
    )

    args = parser.parse_args()

    coordinates = [args.x, args.y, args.z]
    if any(value is None for value in coordinates):
        if not (args.dry_run and all(value is None for value in coordinates)):
            parser.error("--x, --y, --z를 모두 입력하거나 --dry-run만 사용하세요.")

    values = [value for value in coordinates if value is not None]
    values.extend([args.approach, args.lift])

    if not all(math.isfinite(v) for v in values):
        parser.error("좌표는 유한한 숫자여야 합니다.")

    if not (
        0.0 < args.approach <= MAX_STEP
        and 0.0 < args.lift <= MAX_STEP
    ):
        parser.error(
            "approach와 lift는 0보다 크고 0.02m 이하여야 합니다."
        )

    target = coordinates if args.x is not None else None

    # 물체 위쪽 접근점
    pre_pick = (
        [args.x, args.y, args.z + args.approach]
        if target is not None else None
    )

    rclpy.init()
    node = CartesianMove()

    failed = False
    try:
        current = get_xyz(node, diagnose_fk=args.dry_run)

        if args.dry_run and target is None:
            print("\n[DRY RUN] FK 확인 완료. 현재 말단 위치:", current)
            return

        print("\n=== FRONT PICK ===")
        print("현재 위치:", current)
        print("접근 위치:", pre_pick)
        print("Pick 위치:", target)
        print("Lift 높이:", args.lift)
        if args.sequence_test:
            print(
                "[SEQUENCE TEST] 물체 없는 빈 공간 전용. "
                "파지 성공이나 위치 정확도를 보증하지 않습니다."
            )

        if math.dist(current, pre_pick) > MAX_STEP:
            raise RuntimeError(
                "접근점이 현재 위치에서 2cm보다 멉니다. "
                "기존 좌표 이동 프로그램으로 먼저 접근해 주세요."
            )

        if args.dry_run:
            print("\n[DRY RUN] 움직이지 않고 종료")
            return

        client = ActionClient(
            node, GripperCommand, GRIPPER_ACTION
        )

        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                "그리퍼 액션 서버를 찾지 못했습니다."
            )

        # STEP 1: OPEN
        send_gripper(
            node, client, OPEN_POSITION, "OPEN"
        )

        # STEP 2: 접근
        move_xyz(node, pre_pick, "PRE_PICK", args.sequence_test)

        # STEP 3: Pick 위치
        move_xyz(node, target, "PICK", args.sequence_test)

        # STEP 4: CLOSE
        send_gripper(
            node, client, CLOSE_POSITION, "CLOSE"
        )

        # STEP 5: 현재 위치 기준으로 들어 올리기
        actual = get_xyz(node)

        lift_target = [
            actual[0],
            actual[1],
            actual[2] + args.lift,
        ]

        move_xyz(node, lift_target, "LIFT", args.sequence_test)

        final_xyz = get_xyz(node)
        print("최종 말단 좌표:", tuple(round(v, 5) for v in final_xyz))

        print("\n=== PICK 시퀀스 완료 ===")
        print("물체 파지 및 유지 여부는 화면에서 확인하세요.")

    except Exception as e:
        print(f"\n[중단] {e}")
        print("이후 동작은 실행하지 않습니다.")
        failed = True

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
