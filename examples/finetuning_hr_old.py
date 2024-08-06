import json

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 요약해주세요."
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
                return_tensors="pt"
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(
                target,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
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


model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="./model"
)

tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")

tokenizer.pad_token = tokenizer.eos_token

tokenizer.eos_token

train_dataset = CustomDataset("resource/dataset/일상대화요약_train.json", tokenizer)
valid_dataset = CustomDataset("resource/dataset/일상대화요약_dev.json", tokenizer)

from datasets import Dataset

train_dataset = Dataset.from_dict({
    'input_ids': train_dataset.inp,
    'labels': train_dataset.label,
})
valid_dataset = Dataset.from_dict({
    'input_ids': valid_dataset.inp,
    'labels': valid_dataset.label,
})

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

training_args = SFTConfig(
    output_dir='resource/results',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.1,
    num_train_epochs=5,
    max_steps=-1,
    lr_scheduler_type='cosine',
    warmup_steps=20,
    #     log_eval="info",
    #     logging_steps=1,
    save_strategy="epoch",
    save_total_limit=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=1024,
    packing=True,
    seed=42,
)

## Full Fine-tuning
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    args=training_args
)

trainer.train()


## PEFT_Lora
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir='resource/lora_results2',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.1,
    num_train_epochs=5,
    max_steps=-1,
    lr_scheduler_type='cosine',
    warmup_steps=20,
    #     log_eval="info",
    #     logging_steps=1,
    save_strategy="epoch",
    save_total_limit=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=1024,
    packing=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    args=training_args,
    peft_config=lora_config
)

trainer.train()

model.save_pretrained("./resource/lora_results/base/")

## Inference(Full)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

full_model = AutoModelForCausalLM.from_pretrained(
    './resource/results/checkpoint-315/',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
full_model.eval()

## Inference(Lora)

from transformers import AutoModelForCausalLM, TextStreamer
from peft import PeftModel, PeftConfig

peft_model_id = "./resource/lora_results/base/"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="./model"
)
peft_model = PeftModel.from_pretrained(model, peft_model_id)
peft_model.eval()

tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
tokenizer.pad_token = tokenizer.eos_token
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

dataset = CustomDataset("./resource/dataset/일상대화요약_test.json", tokenizer)
with open("./resource/dataset/일상대화요약_test.json", "r") as f:
    result = json.load(f)

import tqdm
for idx in tqdm.tqdm(range(len(dataset))):
    inp = dataset[idx]
    outputs = full_model.generate(
        inp.to("cuda").unsqueeze(0),
        max_new_tokens=1024,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

output_dir = 'base_20240730'
with open(output_dir, "w", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=4))

### Full Inference Results
result = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
print(result)

### Lora Inference Results
result = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
print(result)



