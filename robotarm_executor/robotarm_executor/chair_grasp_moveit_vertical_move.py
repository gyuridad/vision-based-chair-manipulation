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
    "camera_frame": "",
    "tf_base_frame": "World",
    "moveit_base_frame": "world",
    "tip_link": "panda_hand",
    "tf_timeout_sec": 1.0,
    "approach_offset": 0.20,
    "grasp_offset": 0.03,
    "min_goal_z": 0.17,
    "lift_offset": 0.08,
    "retreat_offset": 0.05,
    "open_finger": 0.04,
    "close_finger": 0.0,
    "duration": 1.5,
    "gripper_motion_duration": 0.8,
    "gripper_settle_sec": 0.5,
    "fk_timeout_sec": 2.0,
    "preferred_camera_frame": "",
    "lock_detection_once": True,
    "ignore_stale_when_locked": True,
    "detection_stale_sec": 2.0,
    "orientation_mode": "top_down",
    "fixed_grasp_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "top_down_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
    "orientation_tolerance_rad": 0.12,
    "vertical_move_rate_hz": 40.0,
}

GROUP_NAME = "panda_arm"


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


def apply_gripper(js: JointState, open_width: float):
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        if name in js.name:
            js.position[js.name.index(name)] = open_width


class IKClient(Node):
    def __init__(self):
        super().__init__("chair_grasp_moveit_vertical_move")
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
            vertical_move_rate_hz=max(1.0, float(self.get_parameter("vertical_move_rate_hz").value)),
        )

        self.service_cb_group = ReentrantCallbackGroup()
        self.io_cb_group = ReentrantCallbackGroup()
        self.ik_cli = self.create_client(GetPositionIK, self.args.ik_service, callback_group=self.service_cb_group)
        self.fk_cli = self.create_client(GetPositionFK, self.args.fk_service, callback_group=self.service_cb_group)
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service not available: {self.args.ik_service}")
        if not self.fk_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"FK service not available: {self.args.fk_service}")

        self._latest_js = None
        self._latest_detection = None
        self._locked_detection = None
        self._latest_joint_state_time = None
        self._latest_detection_time = None
        self._executed = False
        self._execution_in_progress = False

        self.create_subscription(
            JointState, self.args.joint_state_topic, self.on_joint_state, 10, callback_group=self.io_cb_group
        )
        self.create_subscription(
            String, self.args.detection_topic, self.on_detection, 10, callback_group=self.io_cb_group
        )
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

        deadline = time.time() + self.args.tf_timeout_sec
        last_exc = None
        while time.time() < deadline:
            for camera_frame in candidates:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.args.tf_base_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=self.args.tf_timeout_sec),
                    )
                    rotation = transform.transform.rotation
                    translation = transform.transform.translation
                    rotmat = quat_xyzw_to_rotmat((rotation.x, rotation.y, rotation.z, rotation.w))
                    trans = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
                    return (rotmat @ point_camera) + trans
                except TransformException as exc:
                    last_exc = exc
            time.sleep(0.02)
        raise RuntimeError(f"TF lookup failed for camera frame candidates {candidates}: {last_exc}")

    def ee_pose_from_fk(self, base_frame=None, tip_link=None, timeout=None):
        if self._latest_js is None:
            raise RuntimeError("No joint state available for FK")

        req = GetPositionFK.Request()
        req.header.frame_id = self.args.moveit_base_frame if base_frame is None else base_frame
        req.fk_link_names = [self.args.tip_link if tip_link is None else tip_link]
        req.robot_state.joint_state = self._latest_js

        future = self.fk_cli.call_async(req)
        timeout = self.args.fk_timeout_sec if timeout is None else timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)
        resp = future.result()
        if resp is None or len(resp.pose_stamped) == 0:
            raise RuntimeError("FK returned empty pose")

        pose = resp.pose_stamped[0].pose
        return SimpleNamespace(
            position=(pose.position.x, pose.position.y, pose.position.z),
            orientation=quat_normalize((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)),
        )

    def compute_ik(self, pos, quat_xyzw, timeout=2.0, avoid_collisions=False, ik_link_name=None, seed_joint_state=None):
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

        seed_source = self._latest_js if seed_joint_state is None else seed_joint_state
        if seed_source is None:
            raise RuntimeError("No joint state available for IK seed")
        seed = RobotState()
        seed.joint_state = seed_source
        req.ik_request.robot_state = seed

        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout + 0.5)
        resp = future.result()
        if resp is None or resp.error_code.val != resp.error_code.SUCCESS:
            raise RuntimeError(f"IK failed: {None if resp is None else resp.error_code.val}")

        solution = resp.solution.joint_state
        solution_map = dict(zip(solution.name, solution.position))
        isaac_order = list(seed_source.name)
        base_map = dict(zip(seed_source.name, seed_source.position))
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

    def move_vertical_axis_locked(
        self,
        from_js,
        fixed_x,
        fixed_y,
        start_z,
        target_z,
        quat_xyzw,
        *,
        duration,
        gripper_width=None,
    ):
        steps = max(1, int(duration * self.args.vertical_move_rate_hz))
        current_js = copy_joint_state(from_js)
        for index in range(1, steps + 1):
            alpha = index / steps
            target_pos = (
                float(fixed_x),
                float(fixed_y),
                float((1.0 - alpha) * start_z + alpha * target_z),
            )
            solved_js = self.compute_ik(target_pos, quat_xyzw, seed_joint_state=current_js)
            if gripper_width is not None:
                apply_gripper(solved_js, gripper_width)
            self.pub.publish(solved_js)
            current_js = copy_joint_state(solved_js)
            self._latest_js = copy_joint_state(solved_js)
            self._latest_joint_state_time = time.time()
            time.sleep(1.0 / self.args.vertical_move_rate_hz)
        return current_js

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
            q_goal_open_seed = self.compute_ik(goal_pos, goal_quat, seed_joint_state=q_pre)
                # 그런데 그냥 계산하지 않고 seed_joint_state=q_pre를 넣었지.
                # 이 뜻은:
                #     "방금 구한 pre 자세를 시작 추정값으로 써서, goal 자세를 더 자연스럽고 가까운 해로 찾자" 는 거야.
                # 왜 seed를 넣냐면:
                #     IK는 해가 여러 개 있을 수 있고
                #     seed를 잘 주면
                #     갑자기 이상한 자세로 꺾이는 것 방지
                #     pre에서 goal로 부드럽게 이어짐
                #     계산 안정성 증가
            apply_gripper(q_goal_open_seed, self.args.open_finger)
            q_lift = self.compute_ik(lift_pos, goal_quat, seed_joint_state=q_goal_open_seed)
            apply_gripper(q_lift, self.args.close_finger)
            q_retreat = self.compute_ik(retreat_pos, goal_quat, seed_joint_state=q_lift)
            apply_gripper(q_retreat, self.args.close_finger)

            current_js = copy_joint_state(self._latest_js)
            self.move_smooth(current_js, q_pre, duration=self.args.duration)

            goal_open_js = self.move_vertical_axis_locked(
                q_pre,
                goal_pos[0],
                goal_pos[1],
                pre_pos[2],
                goal_pos[2],
                goal_quat,
                duration=self.args.duration,
                gripper_width=self.args.open_finger,
            )

            q_close = copy_joint_state(goal_open_js)
            apply_gripper(q_close, self.args.close_finger)
            self.move_smooth(goal_open_js, q_close, duration=self.args.gripper_motion_duration)
            time.sleep(max(0.0, self.args.gripper_settle_sec))
            self.move_smooth(q_close, q_lift, duration=self.args.duration)
            self.move_smooth(q_lift, q_retreat, duration=self.args.duration)

            self._executed = True
            self.get_logger().info(
                "Vertical grasp execution completed from topic detection "
                f"goal={goal_pos}, lift={lift_pos}, retreat={retreat_pos}"
            )
        except Exception as exc:
            self.get_logger().error(f"Vertical grasp execution failed: {exc}")
        finally:
            self._execution_in_progress = False


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
# ros2 run robotarm_executor chair_grasp_moveit_vertical_move
#
# ros2 run robotarm_executor chair_grasp_moveit_vertical_move --ros-args \
#   -p lock_detection_once:=true \
#   -p ignore_stale_when_locked:=true \
#   -p orientation_mode:=top_down \
#   -p min_goal_z:=0.16 \
#   -p grasp_offset:=0.03 \
#   -p lift_offset:=0.08 \
#   -p retreat_offset:=0.05 \
#   -p vertical_move_rate_hz:=40.0
