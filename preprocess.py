import json
import pandas as pd

import torch
from torch.utils.data import Dataset as torchDataset
from datasets import Dataset
from prompt import jh_prompt_template


IGNORE_INDEX = -100



class CustomDataset(torchDataset):
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

        prompt = jh_prompt_template.OBJECT_PROMPT_1(subject_keywords, speaker1, speaker2)
        chat = self.merge_chat_with_prompt(chat_data, prompt)
        message = [
            {'role' : 'system', 'content' : jh_prompt_template.SYSTEM_PROMPT_1},
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
            return_tensors="pt",
            truncation=True, 
            max_length=self.config.max_length,
        )


        input_ids = torch.concat((source.squeeze(0), target['input_ids'].squeeze(0)))
        labels = torch.concat((torch.LongTensor([self.IGNORE_INDEX] * source.shape[1]), target["input_ids"].squeeze(0)))

        return {'input_ids':input_ids, 'labels':labels}
        # return input_ids

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

def lambda_unsqeeze(dataset):
    input_ids = []
    labels = []
    for i in dataset:
        input_ids += [i['input_ids']]
        labels += [i['labels']]

    return {'input_ids':input_ids, 'labels':labels}



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
    

##################################################################################################################################

def collate_fn(examples, tokenizer):
    # input_ids = [example['input_ids'] for example in examples]
    # labels = [example['labels'] for example in examples]
    
    # input_ids = tokenizer.pad({"input_ids": input_ids}, padding="longest", return_tensors="pt")
    # labels = tokenizer.pad({"labels": labels}, padding="longest", return_tensors="pt")
    
    # input_ids = input_ids["input_ids"]
    # labels = labels["labels"]
    
    examples_batch = tokenizer.pad(
        examples,
        padding='longest',
        return_tensors='pt'
    )

    return examples_batch
    # examples_batch = tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    # return examples_batch



class Collate():

    def __init__(self, tokenizer, max_length):
        '''
            tokenizer : hugging face tokeninzer
            max_length : limit sequence length of texts

        '''
        self.tokenizer = tokenizer      # transformers.PreTrainedTokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids = {'input_ids':[i['input_ids'] for i in examples]}
        labels = {'input_ids':[i['labels'] for i in examples]}


        input_ids_batch = self.tokenizer.pad(
            input_ids,
            padding='longest',
            return_tensors='pt',
        )

        label_batch = self.tokenizer.pad(
            labels,
            padding='longest',
            return_tensors='pt',
        )['input_ids']

        input_ids_batch['labels'] = label_batch
        
        return input_ids_batch






# def collate_fn(examples, tokenizer):
#     '''
#     labels를 input ids로 동일하게 할 경우,
#     아닌경우 -> CustomDataset의 것을 그대로 사용.
#            -> 이건 실험을 해봐야 할 듯.
#     '''

    
#     examples_batch = tokenizer.pad(examples['input_ids'], padding="longest", return_tensors="pt")
#     examples_batch['labels'] = examples_batch['input_ids']
#     return examples_batch









##################################################################################################################################

class PreprocessData:

    def __init__(self, path):
        self.df = pd.DataFrame(self.load_data(path))

    def make_columns(self):
        self.df['subject_keyword'] = self.df['input'].apply(lambda x: ', '.join(x['subject_keyword']))
        self.df['conversation'] = self.df['input'].apply(lambda x: x['conversation'])
        self.df['chat_data'] = self.df['conversation'].apply(lambda x: self.lambda_preprocess_chat_format(pd.DataFrame(x)))
        self.df['speaker1'] = self.df['conversation'].apply(lambda x: pd.DataFrame(x)['speaker'].unique()[0])
        self.df['speaker2'] = self.df['conversation'].apply(lambda x: pd.DataFrame(x)['speaker'].unique()[1])
        return self.df    

    def lambda_preprocess_chat_format(self, conv_df:pd.DataFrame) -> str:
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
    

    def load_data(self, fname):
        with open(fname, "r") as f:
            data = json.load(f)
            return data


def preprocess_merge_chat_with_prompt(
        example,
    )->str:
    end_prompt = example['object_chat_prompt']
    chat = example['chat_data']
    out = chat + '\n\n' + end_prompt
    example['chat_with_oject_prompt'] = out
    return example

def preprocess_add_object_prompt(example, user_prompt):
    object_chat_prompt = user_prompt(example['subject_keyword'], example['speaker1'], example['speaker2'])
    example['object_chat_prompt'] = object_chat_prompt
    return example


def preprocess_add_message_prompt(example, system_prompt):
    message = [
        {'role' : 'system', 'content' : system_prompt},
        {'role' : 'user', 'content' : example['object_chat_prompt']}
    ]
    example['system_user_message_prompt'] = message
    return example

def preprocess_make_tokens(example, tokenizer, config):
    
    source = tokenizer.apply_chat_template(
        example['chat_with_oject_prompt'],
        add_generation_prompt=True,
        return_tensors="pt",
    )

    target = example['output']

    target = tokenizer(target,
        return_attention_mask=False,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True, 
        max_length=config.max_length,
    )

    input_ids = torch.concat((source.squeeze(0), target['input_ids'].squeeze(0)))
    labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source.shape[1]), target["input_ids"].squeeze(0)))
    example['input_ids'] = input_ids
    example['labels'] = labels
    return example


def preprocess_to_tokenize(
        train_df:pd.DataFrame,
        tokenizer,
        object_prompt,
        system_prompt,
        config,
    ):

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(lambda x: preprocess_add_object_prompt(x, object_prompt))
    train_dataset = train_dataset.map(lambda x: preprocess_merge_chat_with_prompt(x))
    train_dataset = train_dataset.map(lambda x: preprocess_add_message_prompt(x, system_prompt))
    train_dataset = train_dataset.map(lambda x: preprocess_make_tokens(x, tokenizer, config))
    return train_dataset



if __name__ == "__main__":
    
    class Config:
        max_length = 4096

    config = Config

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B", truncation=True, max_length=4096, padding=True)
    train = PreprocessData('/data1/kaggle/Korean_DCS_2024/data/일상대화요약_train.json')
    train_df = train.make_columns()
    
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(lambda x: preprocess_add_object_prompt(x, jh_prompt_template.OBJECT_PROMPT_1))
    train_dataset = train_dataset.map(lambda x: preprocess_add_message_prompt(x, jh_prompt_template.SYSTEM_PROMPT_1))
    train_dataset = train_dataset.map(lambda x: preprocess_make_tokens(x, tokenizer, config))