import json
import math
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, OrientationConstraint, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from robotarm_common.chair_grasp_common import camera_to_world


DEFAULT_PARAMS = {
    "detection_topic": "/chair_detection_json",
    "joint_state_topic": "/joint_states",
    "joint_command_topic": "/joint_command",
    "ik_service": "/compute_ik",
    "fk_service": "/compute_fk",
    "camera_frame": "",           # “기본값은 비워두고, detection 메시지 안의 frame_id를 우선 쓰겠다”
    "tf_base_frame": "World",     # TF 조회할 때 쓰는 프레임 이름
    "moveit_base_frame": "world", # MoveIt IK/FK 요청할 때 쓰는 기준 프레임 이름
        # 왜 굳이 나눴나?
        #     로봇 시스템에서는 보통 이런 일이 자주 생겨:
        #         TF 트리에서는 World라고 써놓음
        #         MoveIt 설정에서는 world라고 써놓음
    "tip_link": "panda_hand",
    "tf_timeout_sec": 1.0,
    "approach_offset": 0.20,    # 물체를 바로 찍지 않고, 그 위에서 먼저 접근하기 위해 띄우는 거리
    "grasp_offset": 0.03,
    "min_goal_z": 0.17,
    "lift_offset": 0.08,
    "retreat_offset": 0.05,
    "open_finger": 0.04,
    "close_finger": 0.0,
    "duration": 1.5,
    "gripper_motion_duration": 0.8,  # 그리퍼가 open 상태에서 close 상태로 바뀌는 데 쓰는 시간
    "gripper_settle_sec": 0.5,       # 그리퍼를 닫은 뒤 바로 다음 동작으로 넘어가지 않고 0.5초 기다리는 시간
        # 그리퍼를 닫음
        # 바로 lift 하지 않음
        # 0.5초 멈춰서 파지가 안정되도록 기다림
    "fk_timeout_sec": 2.0,
        # 왜 FK만 파라미터화했을 가능성이 크냐면, 이 코드에서는 FK를 현재 EE 자세를 읽는 용도로 쓰고 있고,
        # 이 단계가 막히면 전체 grasp 시작이 지연되니까 조정하기 쉽게 밖으로 뺀 것으로 보여.
        # 반면 IK는 아직 기본 2초로 충분하다고 보고 함수 내부 기본값으로 둔 상태에 가깝다.
    "preferred_camera_frame": "",
        # 코드에서 후보를 모으는 순서를 보면 차이가 딱 보여
        #     for candidate in [
        #         raw_frame,
        #         raw_frame.lstrip("/"),
        #         self.args.preferred_camera_frame,
        #         self.args.camera_frame,
        #     ]:
        # 즉 시도 순서는 이렇게야:
        #     detection 메시지의 camera_info.frame_id
        #     그 frame_id에서 /만 제거한 버전
        #     preferred_camera_frame
        #     camera_frame
        # preferred_camera_frame
        #     “가능하면 이 이름을 우선 써보고 싶다”
        #     detection 메시지 기반 동작을 유지하면서 보조
        # camera_frame
        #     “정 안 되면 이 이름으로라도 해라”
        #     더 강한 fallback
        # 왜 둘 다 빈 문자열일까?
        #     기본값이 둘 다 ""인 이유는
        #     가능하면 detection 메시지 안의 frame_id를 먼저 신뢰하겠다는 뜻이야.

    "lock_detection_once": True,
        # 처음 들어온 유효한 detection 하나를 “고정(lock)”해서,
        # 그 뒤에는 새 detection이 들어와도 계속 그 목표만 쓰게 하는 옵션
        # 첫 유효 detection을 작업 목표로 확정해서,
        # grasp 도중 detection 값이 흔들려도 목표를 바꾸지 않게 하는 옵션
    "ignore_stale_when_locked": True,
        # detection을 한 번 lock한 뒤에는,
        # 그 detection이 시간이 좀 지나서 오래된(stale) 값이 되어도 무시하지 않고 계속 쓰겠다는 뜻
        # 먼저 stale이 뭐냐면
        #     코드에서는 detection 나이를 계산해서:
        #         너무 오래됐으면
        #         그 detection은 믿지 않고 실행 안 하게 돼
        #         기준값은 아래 detection_stale_sec 파라미터
        # ignore_stale_when_locked = True
        #         한번 목표를 정했으면
        #         시간이 좀 지나도
        #         그 목표로 계속 간다
        # ignore_stale_when_locked = False
        #         lock했더라도
        #         시간이 지나서 stale이면
        #         실행 안 한다

    "detection_stale_sec": 2.0,
    "orientation_mode": "top_down",
        # 모드         	 설명       	안정성         사용 추천
        # top_down    	위에서 잡기 	⭐⭐⭐⭐	✅ 기본
        # fixed       	고정 방향   	⭐⭐⭐   	 특정 상황
        # keep_current	현재 방향 유지	 ⭐      	 ❌ 비추천

    "fixed_grasp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "top_down_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "orientation_tolerance_rad": 0.12,
        # 엔드이펙터 방향이 목표 방향에서 최대 약 0.12 rad (≈ 6.8도)까지 틀어져도 허용한다는 의미
        # 🧠 왜 필요한가?
        # IK는 이렇게 생각하면 쉬워:
        #     “이 위치 + 이 방향으로 가는 관절 각도 찾아줘”
        # 근데 방향까지 완벽하게 맞추려면:
        #     해가 없을 수도 있고
        #     계산이 느려질 수도 있음
        # 그래서 tolerance를 줌
}

GROUP_NAME = "panda_arm"
DEFAULT_TF_BASE_FRAME = "World"
DEFAULT_MOVEIT_BASE_FRAME = "world"
DEFAULT_TIP_LINK = "panda_hand"


def quat_normalize(quat):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = quat_normalize(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )

# 원본 JointState 메시지를 안전하게 복사해서 쓰기 위해 필요해.
# 왜 그냥 self._latest_js = msg 하면 안 되나?
# 그걸 그대로 저장하면:
#     다른 곳에서 그 객체를 수정할 수 있고
#     이후 코드에서 손가락 값만 바꾸려다가
#     원본 최신 관절 상태까지 오염될 수 있어
def copy_joint_state(msg: JointState) -> JointState:
    out = JointState()
    out.header = msg.header
    out.name = list(msg.name)
    out.position = list(msg.position)
    out.velocity = list(msg.velocity)
    out.effort = list(msg.effort)
    return out


def quat_from_param(value, default):
    if value is None or len(value) != 4:
        return quat_normalize(default)
    return quat_normalize([float(component) for component in value])


class IKClient(Node):
    def __init__(self):
        super().__init__("chair_grasp_moveit")
        for name, value in DEFAULT_PARAMS.items():
            self.declare_parameter(name, value)
        self.args = SimpleNamespace(
            detection_topic=str(self.get_parameter("detection_topic").value),
            joint_state_topic=str(self.get_parameter("joint_state_topic").value),
            joint_command_topic=str(self.get_parameter("joint_command_topic").value),
            ik_service=str(self.get_parameter("ik_service").value),
            fk_service=str(self.get_parameter("fk_service").value),
            camera_frame=str(self.get_parameter("camera_frame").value),
            tf_base_frame=str(self.get_parameter("tf_base_frame").value),
            moveit_base_frame=str(self.get_parameter("moveit_base_frame").value),
            tip_link=str(self.get_parameter("tip_link").value),
            tf_timeout_sec=float(self.get_parameter("tf_timeout_sec").value),
            approach_offset=float(self.get_parameter("approach_offset").value),
            grasp_offset=float(self.get_parameter("grasp_offset").value),
            min_goal_z=float(self.get_parameter("min_goal_z").value),
            lift_offset=float(self.get_parameter("lift_offset").value),
            retreat_offset=float(self.get_parameter("retreat_offset").value),
            open_finger=float(self.get_parameter("open_finger").value),
            close_finger=float(self.get_parameter("close_finger").value),
            duration=float(self.get_parameter("duration").value),
            gripper_motion_duration=float(self.get_parameter("gripper_motion_duration").value),
            gripper_settle_sec=float(self.get_parameter("gripper_settle_sec").value),
            fk_timeout_sec=float(self.get_parameter("fk_timeout_sec").value),
            preferred_camera_frame=str(self.get_parameter("preferred_camera_frame").value),
            lock_detection_once=bool(self.get_parameter("lock_detection_once").value),
            ignore_stale_when_locked=bool(self.get_parameter("ignore_stale_when_locked").value),
            detection_stale_sec=float(self.get_parameter("detection_stale_sec").value),
            orientation_mode=str(self.get_parameter("orientation_mode").value),
            fixed_grasp_quat_xyzw=[float(v) for v in self.get_parameter("fixed_grasp_quat_xyzw").value],
            top_down_quat_xyzw=[float(v) for v in self.get_parameter("top_down_quat_xyzw").value],
            orientation_tolerance_rad=float(self.get_parameter("orientation_tolerance_rad").value),
        )

        self.service_cb_group = ReentrantCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self.ik_cli = self.create_client(GetPositionIK, self.args.ik_service, callback_group=self.service_cb_group)
        self.fk_cli = self.create_client(GetPositionFK, self.args.fk_service, callback_group=self.service_cb_group)
            # callback_group=self.service_cb_group
            # → 이 클라이언트의 응답 처리를 어떤 콜백 그룹에서 돌릴지 정하는 거야.
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service not available: {self.args.ik_service}")
        if not self.fk_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"FK service not available: {self.args.fk_service}")
            # 🚨 문제 상황 (group 안 나누면)
            #         ROS2 기본은 SingleThreaded처럼 동작하는 경우가 많아서:
            #         상황:
            #             IK 요청 보냄 → 기다림 (blocking 느낌)
            #         그동안:
            #             joint_state 못 받음
            #             detection 못 받음
            #             timer 멈춤
            #         👉 전체 시스템이 “멈춘 것처럼” 보일 수 있음
            # ✅ 그래서 나누는 이유
            #             self.service_cb_group
            #             self.io_cb_group
            #         이렇게 나누면:
            #             서비스 요청 (IK/FK)
            #             센서 입력 / 타이머
            #         👉 서로 독립적으로 실행됨
            # 📊 언제 꼭 나눠야 하냐?
            # ✅ 반드시 나눠야 하는 경우
            #         IK/FK처럼 시간 걸리는 서비스
            #         동시에 여러 콜백이 돌아야 하는 경우
            #         MultiThreadedExecutor 사용 중일 때
            #         로봇 제어 (실시간성 중요)

        self._latest_js = None
        self._latest_detection = None
        self._locked_detection = None
        self._latest_joint_state_time = None
        self._latest_detection_time = None

        self._executed = False               # 이건 작업 완료 후 재실행 방지야.
        self._execution_in_progress = False  # 이건 실행 도중 중복 진입 방지야.
            # 둘 차이를 예시로 보면
            #     실행 전
            #             _execution_in_progress = False
            #             _executed = False
            #         → 실행 가능
            #     실행 시작 직후
            #             _execution_in_progress = True
            #             _executed = False
            #         → 지금 돌고 있으니 중복 실행 금지
            #     실행 성공 후
            #             _execution_in_progress = False
            #             _executed = True
            #         → 이제 안 돌고 있지만, 이미 한 번 끝났으니 재실행 금지

        self.create_subscription(JointState, self.args.joint_state_topic, self.on_joint_state, 10, callback_group=self.io_cb_group)
        self.create_subscription(String, self.args.detection_topic, self.on_detection, 10, callback_group=self.io_cb_group)
        self.pub = self.create_publisher(JointState, self.args.joint_command_topic, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.timer = self.create_timer(0.1, self.try_execute, callback_group=self.io_cb_group)

    def on_joint_state(self, msg: JointState):
        self._latest_js = copy_joint_state(msg)
        self._latest_joint_state_time = time.time()

    def on_detection(self, msg: String):
        try:
            result = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid detection JSON: {exc}")
            return
        if result.get("detection"):
            self._latest_detection = result
            self._latest_detection_time = time.time()
            if self.args.lock_detection_once and self._locked_detection is None:
                self._locked_detection = result

    def _extract_detection_age_sec(self, detection):
        if detection is None or self._latest_detection_time is None:
            return None
        stamp = detection.get("stamp") if isinstance(detection, dict) else None
        try:
            if stamp is not None:
                stamp = float(stamp)
                if stamp >= 1_000_000_000.0:
                    return max(0.0, time.time() - stamp)
        except (TypeError, ValueError):
            pass
        return max(0.0, time.time() - self._latest_detection_time)

    def _resolve_goal_orientation(self, current_orientation):
        mode = (self.args.orientation_mode or "keep_current").strip().lower()
        if mode == "fixed":
            return quat_from_param(self.args.fixed_grasp_quat_xyzw, current_orientation)
        if mode == "top_down":
            return quat_from_param(self.args.top_down_quat_xyzw, current_orientation)
        return quat_normalize(current_orientation)

    def camera_point_to_world(self, detection):
        xyz_camera = detection["detection"].get("xyz_camera")
        if xyz_camera is None:
            raise RuntimeError("Detection does not contain 3D camera coordinates")

        point_camera = np.asarray(xyz_camera, dtype=np.float32)
        t_world_camera = detection.get("t_world_camera")
        if t_world_camera is not None:
            return camera_to_world(point_camera, np.asarray(t_world_camera, dtype=np.float32))

        camera_info = detection.get("camera_info") or {}
        raw_frame = camera_info.get("frame_id") or ""
        candidates = []
        for candidate in [raw_frame, raw_frame.lstrip("/"), self.args.preferred_camera_frame, self.args.camera_frame]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            raise RuntimeError("Detection does not contain T_world_camera or camera frame_id")

        # **“카메라 좌표계의 점(point_camera)을 월드 좌표계 점으로 바꾸기 위해,
        #    가능한 TF 프레임 이름들을 하나씩 시도하는 코드”**야
        base_frame_candidates = []
        # “월드 프레임 이름이 정확히 뭐인지 모르니 후보들을 준비해두자”
        for base_frame in [self.args.tf_base_frame, DEFAULT_TF_BASE_FRAME, "world", "World"]:
            if base_frame and base_frame not in base_frame_candidates:
                base_frame_candidates.append(base_frame)

        last_exc = None   # 이건 마지막으로 실패한 에러를 저장해두는 변수
            # 나중에 모든 시도가 실패했을 때
            # “왜 실패했는지” 알려주려고 남겨두는 거야.
        for base_frame in base_frame_candidates:
            for camera_frame in candidates:   # camera_frame 후보도 하나씩 시도
                try:
                    # “camera_frame에 있는 점을 base_frame 기준으로 바꾸기 위한 TF 변환을 가져와줘”
                    transform = self.tf_buffer.lookup_transform(
                        base_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=self.args.tf_timeout_sec),
                    )
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    rotmat = quat_xyzw_to_rotmat((rotation.x, rotation.y, rotation.z, rotation.w))
                    trans = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
                    # 실제 좌표 변환
                    return (rotmat @ point_camera) + trans
                except TransformException as exc:
                    last_exc = exc
        raise RuntimeError(
            f"TF lookup failed for candidates {candidates} and base frames {base_frame_candidates}: {last_exc}"
        ) from last_exc

    def wait_for_joint_state(self, timeout_sec: float = 5.0):
        start = time.time()
        while time.time() - start < timeout_sec:
            if self._latest_js is not None:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError(f"No {self.args.joint_state_topic} received")

    def ee_pose_from_fk(self, base_frame=None, tip_link=None, timeout=None, joint_state=None):
        joint_state = self._latest_js if joint_state is None else joint_state
        if joint_state is None:
            raise RuntimeError("No joint state available for FK")
        base_frame = self.args.moveit_base_frame if base_frame is None else base_frame
        tip_link = self.args.tip_link if tip_link is None else tip_link

        req = GetPositionFK.Request()
        req.header.frame_id = base_frame
        req.fk_link_names = [tip_link]
        req.robot_state.joint_state = joint_state

        timeout = self.args.fk_timeout_sec if timeout is None else timeout
        future = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        timed_out = not future.done()
        resp = future.result() if future.done() else None
        if resp is None or resp.error_code.val != resp.error_code.SUCCESS:
            if timed_out:
                raise RuntimeError(f"FK timed out after {timeout:.2f}s")
            raise RuntimeError("FK failed")

        pose = resp.pose_stamped[0].pose
        return SimpleNamespace(
            position=(pose.position.x, pose.position.y, pose.position.z),
            orientation=quat_normalize((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)),
        )

    def compute_ik(self, pos, quat_xyzw, timeout=2.0, avoid_collisions=False, ik_link_name=None):
        px, py, pz = [float(v) for v in np.asarray(pos).ravel()[:3]]
        qx, qy, qz, qw = quat_normalize([float(v) for v in np.asarray(quat_xyzw).ravel()[:4]])
        ik_link_name = self.args.tip_link if ik_link_name is None else ik_link_name

        pose = PoseStamped()
        pose.header.frame_id = self.args.moveit_base_frame
        pose.pose.position.x = px
        pose.pose.position.y = py
        pose.pose.position.z = pz
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name = GROUP_NAME
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout.sec = int(timeout)
        req.ik_request.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
        if ik_link_name:
            req.ik_request.ik_link_name = ik_link_name

        constraint = OrientationConstraint()
        constraint.header.frame_id = self.args.moveit_base_frame
        constraint.link_name = ik_link_name
        constraint.orientation = pose.pose.orientation
        constraint.absolute_x_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_y_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.absolute_z_axis_tolerance = self.args.orientation_tolerance_rad
        constraint.weight = 1.0
        constraints = Constraints()
        constraints.orientation_constraints = [constraint]
        req.ik_request.constraints = constraints

        seed = RobotState()
        seed.joint_state = self._latest_js
        req.ik_request.robot_state = seed

        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)
        resp = future.result()
        if resp is None or resp.error_code.val != resp.error_code.SUCCESS:
            raise RuntimeError(f"IK failed: {None if resp is None else resp.error_code.val}")

        solution = resp.solution.joint_state
        solution_map = dict(zip(solution.name, solution.position))
        isaac_order = list(self._latest_js.name)
        base_map = dict(zip(self._latest_js.name, self._latest_js.position))
        base_map.update(solution_map)

        out = JointState()
        out.name = isaac_order
        out.position = [float(base_map.get(name, 0.0)) for name in isaac_order]
        return out

    def move_smooth(self, from_js, to_js, duration=1.5, rate_hz=100):
        names = list(from_js.name)
        if names != list(to_js.name):
            raise RuntimeError("Joint name order mismatch")
        start = np.asarray(from_js.position, dtype=float)
        goal = np.asarray(to_js.position, dtype=float)
        steps = max(1, int(duration * rate_hz))
        for index in range(steps + 1):
            alpha = index / steps
            point = (1.0 - alpha) * start + alpha * goal
            msg = JointState()
            msg.name = names
            msg.position = point.tolist()
            self.pub.publish(msg)
            self._latest_js = copy_joint_state(msg)
            self._latest_joint_state_time = time.time()
            time.sleep(1.0 / rate_hz)

    def try_execute(self):
        if self._executed or self._execution_in_progress or self._latest_js is None:
            return

        detection = self._locked_detection or self._latest_detection
        if detection is None:
            return

        detection_age_sec = self._extract_detection_age_sec(detection)
        if (
            (self._locked_detection is None or not self.args.ignore_stale_when_locked)
            and detection_age_sec is not None
            and detection_age_sec > self.args.detection_stale_sec
        ):
            self.get_logger().warn(
                f"Detection is stale: age={detection_age_sec:.3f}s, limit={self.args.detection_stale_sec:.3f}s"
            )
            return

        self._execution_in_progress = True
        try:
            point_world = self.camera_point_to_world(detection)
            raw_goal_z = float(point_world[2] + self.args.grasp_offset)
            goal_pos = (
                float(point_world[0]),
                float(point_world[1]),
                max(raw_goal_z, self.args.min_goal_z),
            )
            pre_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.approach_offset)
            lift_pos = (goal_pos[0], goal_pos[1], goal_pos[2] + self.args.lift_offset)
            retreat_pos = (goal_pos[0], goal_pos[1], lift_pos[2] + self.args.retreat_offset)

            current_pose = self.ee_pose_from_fk()
            goal_quat = self._resolve_goal_orientation(current_pose.orientation)

            q_pre = self.compute_ik(pre_pos, goal_quat)
            apply_gripper(q_pre, self.args.open_finger)
            q_goal_open = self.compute_ik(goal_pos, goal_quat)
            apply_gripper(q_goal_open, self.args.open_finger)
            q_close = copy_joint_state(q_goal_open)
            apply_gripper(q_close, self.args.close_finger)
            q_lift = self.compute_ik(lift_pos, goal_quat)
            apply_gripper(q_lift, self.args.close_finger)
            q_retreat = self.compute_ik(retreat_pos, goal_quat)
            apply_gripper(q_retreat, self.args.close_finger)

            current_js = copy_joint_state(self._latest_js)
            self.move_smooth(current_js, q_pre, duration=self.args.duration)
            self.move_smooth(q_pre, q_goal_open, duration=self.args.duration)
            self.move_smooth(q_goal_open, q_close, duration=self.args.gripper_motion_duration)
            time.sleep(max(0.0, self.args.gripper_settle_sec))
            self.move_smooth(q_close, q_lift, duration=self.args.duration)
            self.move_smooth(q_lift, q_retreat, duration=self.args.duration)

            self._executed = True
            self.get_logger().info(
                "Grasp execution completed from topic detection "
                f"goal={goal_pos}, lift={lift_pos}, retreat={retreat_pos}"
            )
        except Exception as exc:
            self.get_logger().error(f"Grasp execution failed: {exc}")
        finally:
            self._execution_in_progress = False


def apply_gripper(js: JointState, open_width: float):
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        if name in js.name:
            js.position[js.name.index(name)] = open_width


def main(args=None):
    rclpy.init(args=args)
    node = IKClient()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


### 실행 예시
# ros2 run robotarm_executor chair_grasp_moveit

# ros2 run robotarm_executor chair_grasp_moveit --ros-args \
#   -p lock_detection_once:=true \
#   -p ignore_stale_when_locked:=true \
#   -p orientation_mode:=top_down \
#   -p min_goal_z:=0.16 \
#   -p grasp_offset:=0.03 \
#   -p lift_offset:=0.08 \
#   -p retreat_offset:=0.05