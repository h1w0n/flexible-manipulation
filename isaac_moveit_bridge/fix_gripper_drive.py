import omni.usd
from pxr import UsdPhysics, PhysxSchema

stage = omni.usd.get_context().get_stage()
root = stage.GetRootLayer()

assert root.identifier.endswith(
    "L_corridor_green_box_gripper_test_50.usd"
), "테스트용 USD가 아닙니다."

stage.SetEditTarget(root)

base = "/turtlebot3_manipulation/joints/"
left = stage.GetPrimAtPath(base + "gripper_left_joint")
right = stage.GetPrimAtPath(base + "gripper_right_joint")

left_drive = UsdPhysics.DriveAPI.Get(left, "linear")
assert left_drive, "왼쪽 Drive를 찾지 못했습니다."

# 기존 Mimic 해제
if "PhysxMimicJointAPI:rotY" in right.GetAppliedSchemas():
    right.RemoveAPI(
        PhysxSchema.PhysxMimicJointAPI, "rotY"
    )

# 왼쪽과 동일한 Drive 설정
drive = UsdPhysics.DriveAPI.Apply(right, "linear")

drive.CreateStiffnessAttr(
    left_drive.GetStiffnessAttr().Get()
)
drive.CreateDampingAttr(
    left_drive.GetDampingAttr().Get()
)
drive.CreateMaxForceAttr(
    left_drive.GetMaxForceAttr().Get()
)
drive.CreateTypeAttr(
    left_drive.GetTypeAttr().Get()
)

drive.CreateTargetPositionAttr(0.0)
drive.CreateTargetVelocityAttr(0.0)

print("Schemas:", right.GetAppliedSchemas())
print("Stiffness:", drive.GetStiffnessAttr().Get())
print("Target:", drive.GetTargetPositionAttr().Get())

assert "PhysxMimicJointAPI:rotY" not in right.GetAppliedSchemas()
assert "PhysicsDriveAPI:linear" in right.GetAppliedSchemas()

root.Save()
print("그리퍼 테스트용 USD 저장 완료")