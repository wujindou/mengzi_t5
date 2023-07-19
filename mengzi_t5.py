import numpy as np
import random
import torch
from transformers import set_seed
from dataset import Seq2SeqDataset,DataCollatorForSeq2Seq
import json
set_seed(42)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

train_data = json.load(open('./train_webcpm.json','r',encoding='utf-8'))[:100]
dev_data = json.load(open('./dev_webcpm.json','r',encoding='utf-8'))[:5]
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
model_path = "Langboat/mengzi-t5-base" # huggingface下载模型
Mengzi_tokenizer = T5Tokenizer.from_pretrained(model_path)
Mengzi_model = T5ForConditionalGeneration.from_pretrained(model_path)
trainset = Seq2SeqDataset(train_data)
devset = Seq2SeqDataset(dev_data)
collator = DataCollatorForSeq2Seq(Mengzi_tokenizer)
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
output_dir = "test" # 模型checkpoint的保存目录
training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=1, # batch_size需要根据自己GPU的显存进行设置，2080,8G显存，batch_size设置为2可以跑起来。
        per_device_eval_batch_size=1,
        fp16=True,
        evaluation_strategy="epoch",
        # eval_steps=100,
        load_best_model_at_end=True,
        learning_rate=5e-5,
        #warmup_steps=100,
        output_dir="test",
        save_total_limit=5,
        lr_scheduler_type='constant',
        save_strategy='epoch',
        gradient_accumulation_steps=8,
        dataloader_num_workers=0,
        report_to = 'tensorboard',
        eval_accumulation_steps=2,
)

#

trainer = Trainer(
    tokenizer=Mengzi_tokenizer,
    model=Mengzi_model,
    args=training_args,
    data_collator=collator,
#     data_collator=collator,
    train_dataset=trainset,
    eval_dataset=devset
)

trainer.train()
trainer.save_model("test/best") # 保存最好的模型
