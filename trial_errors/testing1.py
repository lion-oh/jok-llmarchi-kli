from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, Features, Value

class MyTorchDataset(TorchDataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# torch.utils.data.Dataset iterator 생성
torch_dataset = MyTorchDataset([
    {"text": "This is the first example.", "label": 0},
    {"text": "Here's another example.", "label": 1},
    {"text": "And one more example.", "label": 0},
])

def gen(torch_dataset):
    for idx in range(len(torch_dataset)):
        yield torch_dataset[idx]

if __name__ == "__main__":
    print(next(iter(torch_dataset)))
    features = Features({
        "text": Value("string"),
        "label": Value("int32")
    })
    dataset = Dataset.from_generator(gen, gen_kwargs={"torch_dataset": torch_dataset}, features=features)
    print(dataset)