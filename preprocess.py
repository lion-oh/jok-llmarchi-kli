import ast
import json
import pandas as pd

import torch
from torch.utils.data import Dataset
from prompt import prompt_template

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.IGNORE_INDEX=-100
        self.tokenizer = tokenizer
        self.config = config
        self.data = data



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data[idx]

        subject_keywords = row['input']['subject_keyword']
        subject_keywords = ', '.join(subject_keywords)
        conv_df = pd.DataFrame(row['input']['conversation'])
        chat_data = self.preprocess_chat(conv_df)
        speaker1, speaker2 = conv_df['speaker'].unique()

        prompt = prompt_template.USER_PROMPT_1(subject_keywords, speaker1, speaker2)
        chat = self.merge_chat_with_prompt(chat_data, prompt)
        message = [
            {'role' : 'system', 'content' : prompt_template.SYSTEM_PROMPT_1},
            {'role' : 'user', 'content' : chat}
        ]

        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        )


        target = row['output']
        if target != "":
            target += self.tokenizer.eos_token

        target = self.tokenizer(target,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt"
        )

        target['input_ids'] = target['input_ids'].type(torch.int64)

        input_ids = torch.concat((source.squeeze(0), target['input_ids'].squeeze(0)))
        labels = torch.concat((torch.LongTensor([self.IGNORE_INDEX] * source.shape[1]), target["input_ids"].squeeze(0)))

        # return {'input_ids':input_ids, 'labels':labels}
        return input_ids

    def preprocess_chat(self, conv_df:pd.DataFrame)->str:
        result = []
        past_speaker = None
        for _, row in conv_df.iterrows():
            if past_speaker == row['speaker']:
                result += [' ' + row['utterance']]
            else:
                result += [f"\n\n {row['speaker']} : {row['utterance']}"]
                past_speaker = row['speaker']

        xdata = ''.join(result)
        return xdata
    
    def merge_chat_with_prompt(
            self, 
            chat:str, 
            prompt:str
        )->str:
        end_prompt = prompt
        out = chat + '\n\n' + end_prompt
        return out
    


def torchdataset_generator(torch_dataset):
    for idx in range(len(torch_dataset)):
        yield torch_dataset[idx]



##### 밑에 있는 dataCollatorForSupervisedDataset을 사용해서 전처리를 진행할 것인지
##### 혹은 SFTrainer에서 collate fn을 사용해서 전처리를 진행할 것인지,,

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        # 여기서 에러 발생...
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# def collate_fn(examples, tokenizer):
#     '''
#     labels를 input ids로 동일하게 할 경우,
#     아닌경우 -> CustomDataset의 것을 그대로 사용.
#            -> 이건 실험을 해봐야 할 듯.
#     '''

    
#     examples_batch = tokenizer.pad(examples['input_ids'], padding="longest", return_tensors="pt")
#     examples_batch['labels'] = examples_batch['input_ids']
#     return examples_batch


def collate_fn(examples, tokenizer):
    input_ids = [ast.literal_eval(example['input_ids']) for example in examples]
    labels = [ast.literal_eval(example['labels']) for example in examples]
    
    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="longest",
        return_tensors="pt"
    )["input_ids"]
    
    labels = torch.tensor(labels)
    
    return {"input_ids": input_ids, "labels": labels}