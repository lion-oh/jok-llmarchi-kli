# pip install rouge
# pip install git+https://github.com/google-research/bleurt.git
import numpy as np
from rouge import Rouge
import datasets


BLEURT_METRIC = datasets.load_metric("bleurt")
ROUGE = Rouge()

# model_out = ["he began by starting a five person war cabinet and included chamberlain as lord president of the council",
#              "the siege lasted from 250 to 241 bc, the romans laid siege to lilybaeum",
#              "the original ocean water was found in aquaculture"]

# reference = ["he began his premiership by forming a five-man war cabinet which included chamberlain as lord president of the council",
#              "the siege of lilybaeum lasted from 250 to 241 bc, as the roman army laid siege to the carthaginian-held sicilian city of lilybaeum",
#              "the original mission was for research into the uses of deep ocean water in ocean thermal energy conversion (otec) renewable energy production and in aquaculture"]
# rouge = Rouge()
# rouge.get_scores(model_out, reference, avg=True)


# def compute_metrics(pred):
#     references = pred.label_ids
#     generated_texts = pred.predictions
    
#     bleu_scores = []
#     for reference, generated_text in zip(references, generated_texts):
#         reference_text = train_dataset[reference]['text']
#         bleu_score = sentence_bleu([reference_text], generated_text)
#         bleu_scores.append(bleu_score)

#     return {
#         'bleu': sum(bleu_scores) / len(bleu_scores)
#     }


### 아래 reference를 바탕으로 코드 만들기.
# https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99


def compute_rogue(pred):

    references = pred.labels
    generated_texts = pred.predictions
    rouge1_scores = []
    for reference, generated_text in zip(references, generated_texts):
        idx = np.where(reference.to_numpy() != -100)[0]                  # ignore index = -100
        reference = reference[idx:]
        generated_text = generated_text[idx:]
        output = ROUGE.get_score(generated_text, reference)
        rouge1 = output['rouge-1']
        rouge1_f1, rouge1_p, rouge1_r = rouge1['f'], rouge1['p'], rouge1['r']
        rouge1_scores += [rouge1_f1]

    return {
        'rouge' : rouge1_scores
    }


def compute_bleurt(pred):
    references = pred.labels
    generated_texts = pred.predictions
    bleurt_scores = []
    for reference, generated_text in zip(references, generated_texts):
        idx = np.where(reference.to_numpy() != -100)[0]                  # ignore index = -100
        reference = reference[idx:]
        generated_text = generated_text[idx:]
        output = BLEURT_METRIC.compute(predictions=generated_text, references=reference)
        bleurt_scores += output # output은 리스트임.

    return {
        'bleurt' : bleurt_scores
    }


def compute_metric(pred):
    rouge = compute_rogue(pred)
    bleurt = compute_bleurt(pred)
    rouge.update(bleurt)
    return rouge


if __name__ == "__main__":
    aa = datasets.load_metric('bleurt')
    print(aa.inputs_description)