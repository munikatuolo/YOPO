import argparse
import os

import rospy

from core.drone import YopoDrone
from core.environment import FlightEnvironment


class YopoTesting:
    def __init__(self, settings, weight_path, drone_id="drone_0", comm_range=20.0):
        self.settings = settings
        self.weight_path = weight_path
        self.drone_id = drone_id
        self.comm_range = comm_range

    def run(self):
        rospy.init_node("yopo_net", anonymous=False)
        env = FlightEnvironment(self.settings)
        drone = YopoDrone(
            environment=env,
            weight_path=self.weight_path,
            drone_id=self.drone_id,
            comm_range=self.comm_range,
            use_tensorrt=bool(self.settings["use_tensorrt"]),
        )
        rospy.sleep(1.0)
        rospy.Timer(rospy.Duration(drone.ctrl_dt), drone.control_pub_callback)
        print("YOPO Net Node Ready!")
        rospy.spin()


class TestingConfig:
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
        parser.add_argument("--trial", type=int, default=1, help="trial number")
        parser.add_argument("--epoch", type=int, default=50, help="epoch number")
        parser.add_argument("--drone_id", type=str, default="drone_0", help="unique drone id")
        parser.add_argument("--comm_range", type=float, default=20.0, help="communication range")
        return parser

    @staticmethod
    def resolve_weight(args):
        base_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
        base_dir = os.path.abspath(base_dir)
        return "yopo_trt.pth" if args.use_tensorrt else f"{base_dir}/saved/YOPO_{args.trial}/epoch{args.epoch}.pth"
