import json
import time
from threading import Lock

import cv2
import numpy as np
import rospy
import std_msgs.msg
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

from config.config import cfg
from control_msg import PositionCommand
from policy.poly_solver import LatticePrimitive, Poly5Solver, Polys5Solver, calculate_yaw
from policy.state_transform import StateTransform
from policy.yopo_network import YopoNetwork

try:
    from torch2trt import TRTModule
except ImportError:
    TRTModule = None


class DroneCommunication:
    """Publish/subscribe drone states and keep neighbors in communication range."""

    def __init__(self, drone_id, max_range=20.0, topic="/yopo_net/drone_states"):
        self.drone_id = drone_id
        self.max_range = max_range
        self.topic = topic
        self.neighbor_states = {}
        self.publisher = rospy.Publisher(self.topic, String, queue_size=10)
        self.subscriber = rospy.Subscriber(self.topic, String, self._on_state, queue_size=10)

    def publish_state(self, position, velocity):
        payload = {
            "id": self.drone_id,
            "position": np.asarray(position, dtype=float).tolist(),
            "velocity": np.asarray(velocity, dtype=float).tolist(),
            "stamp": rospy.Time.now().to_sec(),
        }
        self.publisher.publish(String(data=json.dumps(payload)))

    def _on_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        peer_id = data.get("id")
        if peer_id is None or peer_id == self.drone_id:
            return
        self.neighbor_states[peer_id] = data

    def get_neighbors_in_range(self, self_position):
        nearby = {}
        self_position = np.asarray(self_position, dtype=float)
        for peer_id, state in self.neighbor_states.items():
            peer_pos = np.asarray(state.get("position", [np.inf, np.inf, np.inf]), dtype=float)
            if np.linalg.norm(peer_pos - self_position) <= self.max_range:
                nearby[peer_id] = state
        return nearby


class YopoDrone:
    def __init__(self, environment, weight_path, drone_id="drone_0", comm_range=20.0, use_tensorrt=False):
        cfg["train"] = False
        self.environment = environment
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = self.environment.goal.copy()
        self.plan_from_reference = self.environment.plan_from_reference
        self.use_trt = use_tensorrt
        self.verbose = self.environment.verbose
        self.visualize = self.environment.visualize
        self.rotation_bc = R.from_euler("ZYX", [0, self.environment.pitch_angle_deg, 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.rotation_wc = np.eye(3)
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30

        if self.use_trt:
            if TRTModule is None:
                raise RuntimeError("torch2trt is required when --use_tensorrt=1")
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight_path))
        else:
            state_dict = torch.load(weight_path, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        self.communication = DroneCommunication(drone_id=drone_id, max_range=comm_range)

        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.environment.ctrl_topic, PositionCommand, queue_size=1)

        self.odom_sub = rospy.Subscriber(self.environment.odom_topic, Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.environment.depth_topic, Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)

    def callback_set_goal(self, data):
        self.goal = self.environment.update_goal(data.pose.position.x, data.pose.position.y)
        self.arrive = False

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]).as_euler("ZYX", degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True
        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        self.communication.publish_state(pos, self.desire_vel if self.desire_vel is not None else np.zeros(3))
        _ = self.communication.get_neighbors_in_range(pos)
        if np.linalg.norm(pos - self.goal) < 5 and not self.arrive:
            self.arrive = True

    def process_odom(self):
        rotation_wb = R.from_quat([
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ]).as_matrix()
        self.rotation_wc = np.dot(rotation_wb, self.rotation_bc)
        rotation_cw = self.rotation_wc.T

        vel_w = self.desire_vel if self.plan_from_reference else np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ])
        vel_c = np.dot(rotation_cw, vel_w)
        acc_c = np.dot(rotation_cw, self.desire_acc)
        goal_c = np.dot(rotation_cw, self.goal - self.desire_pos)
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        return self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init:
            return
        time0 = time.time()
        if data.encoding == "32FC1":
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        depth = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0
        depth = depth.reshape([1, 1, self.height, self.width])

        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(self.process_odom().to(self.device, non_blocking=True))
        time2 = time.time()

        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.rotation_wc, endstate_c)

        action_id = np.argmin(score) if self.visualize else 0
        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0], endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1], endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2], endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub_callback(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            yaw, yaw_dot = calculate_yaw(self.desire_vel, self.goal - self.desire_pos, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)
        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num - 1, -1, -1))
        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((self.optimal_poly_x.get_position(t_values), self.optimal_poly_y.get_position(t_values), self.optimal_poly_z.get_position(t_values)), axis=-1)
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = np.dot(self.lattice_primitive.lattice_pos_node.cpu().numpy(), self.rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0], lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1], lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2], lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((lattice_poly_x.get_position(t_values), lattice_poly_y.get_position(t_values), lattice_poly_z.get_position(t_values)), axis=-1)
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            self.lattice_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0], pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1], pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2], pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((all_poly_x.get_position(t_values), all_poly_y.get_position(t_values), all_poly_z.get_position(t_values)), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, points_array))

    def print_time(self, time0, time1, time2, time3, time4, time5):
        self.time_interpolation += (time1 - time0)
        self.time_prepare += (time2 - time1)
        self.time_forward += (time3 - time2)
        self.time_process += (time4 - time3)
        self.time_visualize += (time5 - time4)
        self.count += 1
        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if self.verbose or total_time > tolerance:
            print(
                f"Average(ms): interp={1000 * self.time_interpolation / self.count:.2f}, "
                f"prepare={1000 * self.time_prepare / self.count:.2f}, "
                f"forward={1000 * self.time_forward / self.count:.2f}, "
                f"post={1000 * self.time_process / self.count:.2f}, "
                f"visual={1000 * self.time_visualize / self.count:.2f}"
            )

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(torch.zeros((1, 9), dtype=torch.float32, device=self.device))
        endstate_pred, _ = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)
