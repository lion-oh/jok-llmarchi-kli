import os
from dataclasses import dataclass

@dataclass
class base_dataset:
    dataset: str = "competition_dataset"
    train_split: str = os.path.join(os.path.dirname(__file__), "../../baseline/resource/data/일상대화요약_train.json")
    valid_split: str = os.path.join(os.path.dirname(__file__), "../../baseline/resource/data/일상대화요약_dev.json")
    test_split: str = os.path.join(os.path.dirname(__file__), "../../baseline/resource/data/일상대화요약_test.json")