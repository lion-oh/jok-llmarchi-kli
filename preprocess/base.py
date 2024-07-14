'''학습 데이터 형태 출력'''
import json
from typing import (
    List
)



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



if __name__ == "__main__":
    # fileName = '일상대화요약_train.json'
    fileName = 'sample.json'
    f_path = f'../baseline/resource/data/{fileName}'
    with open(f_path, "r") as file:
        data = json.load(file)
    pprint_data(data)
