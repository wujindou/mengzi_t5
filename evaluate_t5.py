from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import json
model_path = "Langboat/mengzi-t5-base" # huggingface下载模型
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
import torch
device = torch.device('cuda')
model.to(device)

best_precision = 0
best_beam = 0
preds = []
def inference(model,inputs,max_s_length=1024,max_target_length=256):
    padding=True
    model_inputs = tokenizer(inputs,max_length=max_s_length, padding=padding, truncation=True,return_tensors='pt').input_ids
    outputs= model.generate(model_inputs.to(device),max_length=max_target_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def inference_with_file(dataset,batch_size=8):
    results =[]
    batch_data = []
    for idx,data in enumerate(dataset):
      batch_data.append('请根据问题和文章从文章中抽取回答问题的相关内容并添加引用文章。输入'+data['question']+';文章：'+data['context'])
      if len(batch_data)==batch_size:
          # print(batch_data)
          # sys.exit(1)
          batch_preds = inference(model,batch_data)
          batch_data = []
          results.extend(batch_preds)
    if len(batch_data)>0:
      batch_preds = inference(model,batch_data)
      results.extend(batch_preds)
    return results

dev_data = json.load(open('./dev_webcpm.json','r',encoding='utf-8'))
preds = inference_with_file(dev_data)
targets = [d['answer'] for d in dev_data]
print(preds[0]+'\t'+targets[0])
from rouge import Rouge
rouge = Rouge()

def rouge_score(candidate, reference):
  text1 = " ".join(list(candidate))
  text2 = " ".join(list(reference))
  score = rouge.get_scores(text1, text2)
  return score


import sys
sys.setrecursionlimit(8735 * 2080 + 10)
r1 = []
r2 = []
R_L= []
error_cnt =0
for p,t in zip(preds,[' '.join(list(t)) for t in targets]):
  try:
    scores = rouge_score(p,t)
    r1.append(scores[0]["rouge-1"]["f"])
    r2.append(scores[0]["rouge-2"]["f"])
    R_L.append(scores[0]["rouge-l"]["f"])
  except Exception as e:
    error_cnt+=1
    pass
rouge_1 = sum(r1)/len(r1)
rouge_2 = sum(r2)/len(r2)
rouge_l = sum(R_L)/len(R_L)
print('rouge-1 :'+str(rouge_1)+' rouge-2 :'+str(rouge_2)+' rouge-l :'+str(rouge_l))
print(error_cnt)
