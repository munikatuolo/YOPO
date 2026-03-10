from core.testing import TestingConfig, YopoTesting


if __name__ == "__main__":
    args = TestingConfig.parser().parse_args()
    weight = TestingConfig.resolve_weight(args)
    print("load weight from:", weight)

    settings = {
        "use_tensorrt": args.use_tensorrt,
        "goal": [50, 0, 2],
        "pitch_angle_deg": 0,
        "odom_topic": "/sim/odom",
        "depth_topic": "/depth_image",
        "ctrl_topic": "/so3_control/pos_cmd",
        "plan_from_reference": False,
        "verbose": False,
        "visualize": True,
    }

    YopoTesting(settings, weight_path=weight, drone_id=args.drone_id, comm_range=args.comm_range).run()
