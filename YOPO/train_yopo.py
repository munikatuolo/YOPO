from core.training import TrainingConfig, YopoTraining


if __name__ == "__main__":
    args = TrainingConfig.parser().parse_args()
    YopoTraining(pretrained=args.pretrained, trial=args.trial, epoch=args.epoch).run()
    print("Run YOPO Finish!")
