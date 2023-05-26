import torch
import json
import copy
from torch.utils.data import Dataset

def load_data(path):
    with open(path, "r") as fp:
        data = fp.read().strip().split("\n") 
    return data


def print_dataset_example(input_input_ids, label_input_ids, tokenizer):
    print("input_ids",input_input_ids)
    print("input_tokens", tokenizer.convert_ids_to_tokens(input_input_ids))
    print("inputs", tokenizer.decode(input_input_ids))
    print("label_ids", label_input_ids)
    print("label_tokens", tokenizer.convert_ids_to_tokens(label_input_ids))
    print("labels", tokenizer.decode(label_input_ids))

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

IGNORE_INDEX = -100
    
class NerCollate:
    def __init__(self, args, tokenizer):
        self.instruct_column = args.instruct_column
        self.query_column = args.query_column
        self.response_column = args.response_column
        self.history_column = None
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        
    def collate_fn(self, batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for example in batch:
            if isinstance(example, str):
                example = json.loads(example)
            prompt = 'Human: \n' + example[self.instruct_column] + example[self.query_column] + '\n\nAssistant: \n' 
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # do not add bos_token_id
            prompt_label = [IGNORE_INDEX] * len(prompt_ids)
            response = example[self.response_column]
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)  # do not add bos_token_id
            response_ids = [self.tokenizer.bos_token_id] + response_ids
            response_label = [-100] + copy.deepcopy(response_ids)
            input_ids = prompt_ids + response_ids
            labels = prompt_label + response_label
            # add eos at every end of assistant sentence
            input_ids += [self.tokenizer.eos_token_id] # make sure eos_token_id is correct
            labels += [self.tokenizer.eos_token_id]
            
            # print(input_ids)
            # print(labels)
            input_ids = input_ids[:self.max_seq_length-2]
            labels = labels[:self.max_seq_length-2]
            attention_mask = [1] * len(input_ids)
            input_ids = [self.tokenizer.pad_token_id] * (self.max_seq_length-len(input_ids)) + input_ids
            labels = [IGNORE_INDEX] * (self.max_seq_length-len(labels)) + labels
            attention_mask = [0] * (self.max_seq_length-len(attention_mask)) + attention_mask
            assert len(input_ids) == len(labels) == len(attention_mask) == self.max_seq_length
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
        results = {'input_ids': torch.tensor(all_input_ids), 'labels': torch.tensor(all_labels), "attention_mask": torch.tensor(all_attention_mask)}
        return results
    
if __name__ == "__main__":
  class Args:
    max_seq_length = 128+64
    instruct_column = "instruct"
    query_column = "query"
    response_column = "answer"
    train_path = "data/msra/instruct_data/train.txt"

  args = Args()
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("./model_hub/BELLE-7B-2M")
  data = load_data(args.train_path)[:10]
  data = [
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。", "answer": "郑振铎_人名\n阿英_人名\n国民党_机构名"},
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。", "answer": "北京市_地名"},
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。", "answer": "姜德明_人名"},
    ]
  data = [data[1]]
  print(data)

  ner_collate = NerCollate(args, tokenizer)
  
  from torch.utils.data import DataLoader
  train_dataloader = DataLoader(data,
                  batch_size=1,
                  shuffle=False,
                  drop_last=True,
                  num_workers=0,
                  collate_fn=ner_collate.collate_fn)
  for step, batch in enumerate(train_dataloader):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print(input_ids.shape, labels.shape)
    break

  # train_dataset = ner_collate.collate_fn(data) 
  # print(train_dataset["input_ids"][0])
    
  print(tokenizer.decode([52828, 66, 100389, 2]))
