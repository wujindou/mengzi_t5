#coding:utf-8
import sys
import json
class Seq2SeqDataset:
  def __init__(self, data):
      self.datas = data

  def __len__(self):
      return len(self.datas)

  def __getitem__(self, index):
      return self.datas[index]

class DataCollatorForSeq2Seq:
  def __init__(self, tokenizer, padding: bool = True, max_length: int = 128):
      self.tokenizer = tokenizer
      #self.model = model
      self.max_source_length = 1024
      self.max_target_length = 512
      self.padding = padding
      self.max_length = max_length

  def __call__(self, batch):
      features = self.collator_fn(batch)
      return features


  def preprocess(self, item):
      source = '请根据问题和文章从文章中抽取回答问题的相关内容并添加引用文章。输入'+item['question']+';文章：'+item['context'].replace(' ','')+'##回答:'
      target = item["answer"]
      return source, target

  def collator_fn(self, batch):
      results = map(self.preprocess, batch)
      inputs, targets = zip(*results)

      input_tensor = self.tokenizer(inputs,
                                    truncation=True,
                                    padding=True,
                                    max_length=self.max_source_length,
                                    return_tensors="pt",
                                    )

      target_tensor = self.tokenizer(targets,
                                      truncation=True,
                                      padding=True,
                                      max_length=self.max_target_length,
                                      return_tensors="pt",
                                      )

      input_tensor["labels"] = target_tensor["input_ids"]

      if "token_type_ids" in input_tensor:
          del input_tensor["token_type_ids"]
      return input_tensor
