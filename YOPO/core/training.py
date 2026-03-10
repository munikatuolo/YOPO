import argparse
import os
import random

import numpy as np
import torch

from policy.yopo_trainer import YopoTrainer


class YopoTraining:
    def __init__(self, pretrained=0, trial=1, epoch=50):
        self.pretrained = pretrained
        self.trial = trial
        self.epoch = epoch

    @staticmethod
    def configure_random_seed(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def run(self):
        self.configure_random_seed(0)
        log_dir = os.path.dirname(os.path.abspath(__file__)) + "/../saved"
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_path = f"{log_dir}/YOPO_{self.trial}/epoch{self.epoch}.pth" if self.pretrained else ""
        trainer = YopoTrainer(
            learning_rate=1.5e-4,
            batch_size=16,
            loss_weight=[1.0, 1.0],
            tensorboard_path=log_dir,
            checkpoint_path=checkpoint_path,
            save_on_exit=True,
        )
        trainer.train(epoch=50)


class TrainingConfig:
    @staticmethod
    def parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
        parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
        parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
        return parser
