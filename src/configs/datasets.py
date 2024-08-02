from dataclasses import dataclass


@dataclass
class base_dataset:
    dataset: str="competition_dataset"
    train_split: str="../../baseline/resource/data/일상대화요약_train.json"
    valid_split: str="../../baseline/resource/data/일상대화요약_dev.json"
