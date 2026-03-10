import numpy as np


class FlightEnvironment:
    """Configure and store drone flight environment settings."""

    def __init__(self, settings):
        self.settings = settings
        self.goal = np.array(settings["goal"], dtype=float)
        self.odom_topic = settings["odom_topic"]
        self.depth_topic = settings["depth_topic"]
        self.ctrl_topic = settings["ctrl_topic"]
        self.pitch_angle_deg = settings["pitch_angle_deg"]
        self.plan_from_reference = settings["plan_from_reference"]
        self.verbose = settings["verbose"]
        self.visualize = settings["visualize"]

    def update_goal(self, x, y, z=2.0):
        self.goal = np.array([x, y, z], dtype=float)
        return self.goal
