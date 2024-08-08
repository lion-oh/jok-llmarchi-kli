import json
from typing import List

import torch
from torch.utils.data import Dataset

from src.prompt.templates.base import _PROMPT_PREFIX


def pprint_data(data: List, k: int = 3, n: int = 5):
    data = data[:k]
    for sample in data:
        id = sample.get('id')
        conversations = sample['input'].get('conversation')[:n]
        subject_keyword = sample['input'].get('subject_keyword')
        outputs = sample['output']
        print(f"\n**** Conversation ID: {id} ****\n")
        print("INPUTS:")
        print("\t<대화내용>")
        for cvt in conversations:
            speaker = cvt.get('speaker')
            utterance = cvt.get('utterance', '')
            print(f'\t- 화자({speaker}): {utterance}')
        print('\t- ...')
        print(f'\t<핵심 키워드>: {subject_keyword[0]}\n')
        print(f"OUTPUT: {outputs}\n\n")


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []

        PROMPT = _PROMPT_PREFIX

        with open(fname, "r") as f:
            data = json.load(f)

        # 화자 중복제거, 대화 앞뒤로 [Conversation], [Question] 붙이기
        def make_chat(inp):
            chat = []
            current_chat = []
            current_speaker = None
            current_utterance = ''
            
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                
                if speaker == current_speaker:
                    current_utterance += ' ' + utterance
                
                else:
                    if current_speaker is not None:
                        current_chat.append(f'화자{current_speaker}: {current_utterance}')
                    current_speaker = speaker
                    current_utterance = utterance
                    
            if current_utterance:
                current_chat.append(f'화자{current_speaker}: {current_utterance}')

            chat = '\n'.join(current_chat)
            chat = '[Conversation]\n' + chat
            keyword = ', '.join(inp['subject_keyword'])
            question = f"[Question]\n위 {keyword} 주제에 대한 대화를 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]

            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(target,
                               return_attention_mask=False,
                               add_special_tokens=False,
                               return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True,
                                                 padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


if __name__ == "__main__":
    '''Print Sample Data'''
    fileName = 'sample.json'
    PATH = f'../../baseline/resource/data/{fileName}'
    with open(PATH, "r") as file:
        data = json.load(file)
    pprint_data(data)