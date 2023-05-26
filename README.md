# Chinese-BELLE-LoRA-Tuning
使用LoRA对BELLE发布的BELLE-7B-2M进行微调。整体的结构非常简单，构造好相应格式的数据后就可以开始训练。

BELLE-7B-2M下载地址：[BelleGroup/BELLE-7B-2M at main (huggingface.co)](https://huggingface.co/BelleGroup/BELLE-7B-2M/tree/main)

训练好的实体识别LoRA权重已经位于checkpoint下。

# 依赖

linux操作系统为Ubantu，GPU为A40-48G显存。

```python
mpi4py
transformers==4.28.1
peft==0.3.0
icetk
deepspeed==0.9.2
accelerate
cpm_kernels
sentencepiece==0.1.99
peft=0.3.0
torch=2.0.0 
```

# 说明

## 目录结构

```python
--checkpoint：保存模型
----msra：数据集名称
--------train_deepspeed
------------adapter_model
----------------adapter_config.json
----------------adapter_model.bin
----------------train_args.json
--------train_trainer
------------adapter_model
----------------adapter_config.json
----------------adapter_model.bin
----------------train_args.json
--model_hub：预训练模型
----BEELE-7B-2M：预训练模型位置
--data：数据
----msra：数据集名称
--------instruct_data：指令数据
------------dev.txt
------------train.txt
--------ori_data：原始数据
--chat_ner.py：闲聊
--train_deepspeed.py：使用原生deepspeed训练
--train_trainer.py： 使用transformers的Trainer进行训练
--test.py：测试训练好的模型
--predict.py：预测
--process.py：处理数据为instruct_data
--dataset.py：加载数据为相应的格式
--deepspeed.json：deepspeed配置文件，用于trasnformers的Trainer
--config_utils.py：用于用字典定义配置，并接收命令行参数
```

## 数据格式

这里我们以命名实体识别任务为例，数据在data/msra下，其中ori_data为原始数据,instruct_data为处理后的数据，数据格式为一条一个样本，具体是：

```python
{"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。", "answer": "日_地名\n京_地名\n北京_地名"}
```

可以按照自己的任务自行构建。

## 一般过程

1、data下新建数据集，用process.py处理数据为instruct_data下的数据。

2、这里使用train_trainer.py进行训练，为了能够让transformers的Trainer在训练的过程中保存lora权重，对Trainer进行相应的修改，参考：https://github.com/huggingface/peft/issues/96 。因为有了config_utils.py，我们可以在字典里面定义相关参数，然后可以在命令行修改桉树的值（嵌套参数之间用_分隔）。

```python
args = {
    "data_name": "msra",  # 数据集名称
    "model_dir": "/root/autodl-tmp/chatglm-6b/",  # chatglm-6b地址，修改为自己的路径
    "lora_r": 8,  # lora参数
    "max_source_length": 128,  # instruct+query的最大长度
    "max_target_length": 32,  # answer的最大长度
    "instruct_column": "instruct",  # instruct列名
    "query_column": "query",  # query列名
    "response_column": "answer",  # answer列名
    "train_path": "data/msra/instruct_data/train.txt", # 训练数据，修改为自己数据
    "dev_path": "data/msra/instruct_data/dev.txt",  # 测试数据，修改为自己数据
    "ignore_pad_token_for_loss": True,  # 默认就好
    "train_batch_size": 12,  # 训练batch_size
    "gradient_accumulation_steps": 1,  # 默认就好
    "save_dir": "/root/autodl-tmp/msra_trainer/",  # 保存模型位置，修改为自己的路径
    "num_train_epochs": 1,  # 训练epoch
    "local_rank": -1,  # deepspeed所需，默认就好
    "log_steps": 10,  # 多少步打印一次结果
    "save_steps": 50,  # 多少步保存一次模型
    "deepspeed_json_path": "deepspeed.json" # deepspeed配置
}
```

需要注意的是，Trainer中使用deepspeed要保持deepspeed定义的参数和Trainer里面参数保持一致，比如：deepspeed.json：

```python
{
  "train_micro_batch_size_per_gpu": 12,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-05,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-08,
      "weight_decay": 0.0005
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000.0,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000.0,
    "contiguous_gradients": true
  }
}
```

- train_micro_batch_size_per_gpu和per_device_train_batch_size
- lr和learning_rate
- betas里面和adam_beta1、adam_beta2
- weight_decay和weight_decay
- fp16和fp16

默认的话不用修改这些。

## 训练

```python
deeepspeed train_deepspeed.py 或者 deepspeed train_trainer.py
```

## 测试

修改data_name，运行：`python test.py`

```python
预测： ['莫高窟_地名\n敦煌__地名\n', '时城出版社_机构名\n', '中国_地名\n东京国际法学会_机构名\n', '庄_人名\n庄__人名\n庄周_人名\n', '玉峰_地名\n重庆_地名\n', '西柏坡_地名\n', '没有_地名\n中国共产党_机构名\n', '宝顶山_地名\n七宗史机构\n', '深_地名\n', '瑞士_地名\n西班牙_地名\n比利时_地名\n丹麦_地名\n', '巫峡_地名\n', '卫生部_机构名\n黄黄骅市中医精神病专科医院_机构名\n', '京沪高速铁路_地名\n', '中国_地名\n', '西柏坡_地名\n阎会强_人名\n王二刚_人名\n库力辉_人名\n阎东霞_人名\n阎明明_人名\n阎长明_人名\n', '东方东方艺术》杂志_机构名\n越秀_机构名\n', '没有_地名\n', '蚩尤寨_地名\n龙王塘_地名\n', '宋神宗_人名\n杨_机构名\n杨次公_人名\n', '敦煌_地名\n莫高窟_地名\n']

真实： ['莫高窟_地名\n敦煌城_地名', '花城出版社_机构名', '中国_地名\n东京国际法学会_机构名', '祝_人名\n庄子_人名\n庄周_人名', '玉峰_地名\n重庆_地名', '西柏坡_地名', '中国_地名\n中国共产党_机构名', '宝顶山_地名\n密宗_人名', '深_地名', '瑞士_地名\n西班牙_地名\n比利时_地名\n丹麦_地名', '巫峡_地名', '卫生部_机构名\n河北黄骅市中医精神病专科医院_机构名', '京沪高速铁路_地名', '中国_地名', '西柏坡村_地名\n阎会强_人名\n王二刚_人名\n刘力辉_人名\n阎东霞_人名\n阎明明_人名\n阎长明_人名', '《东方艺术》杂志_机构名\n越秀_机构名', '纪念馆_地名', '蚩尤寨_地名\n龙王塘村_地名', '宋神宗_人名\n礼部_机构名\n杨次公_人名', '敦煌_地名\n莫高窟_地名']
```

## 预测

修改data_name，运行：`python predict.py`

```python
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。"
预测 >>>  郑振铎_人名
阿英_人名
解放_机构名

真实 >>>  郑振铎_人名
阿英_人名
国民党_机构名
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。"
预测 >>>  北京市_地名

真实 >>>  北京市_地名
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。"
预测 >>>  姜德明_人名

真实 >>>  姜德明_人名
```

## 闲聊

修改data_name，运行：`python chat_ner.py --model_name "bloom" --base_model "./model_hub/BELLE-7B-2M" --tokenizer_path "./model_hub/BELLE-7B-2M" --lora_model "./checkpoint/msra/train_trainer/adapter_model" --with_prompt --interactive`

````python
加载模型耗时：5.13695338567098分钟
loading peft model
Start inference with instruction mode.
=====================================================================================
+ 当前使用的模型是：bloom
-------------------------------------------------------------------------------------
+ 该模式下仅支持单轮问答，无多轮对话能力。
+ 如果是llama或者alpaca模型，如要进行多轮对话，请使用llama.cpp或llamachat工具。
=====================================================================================
Input:你好

Assistant: 
你好，有什么需要帮助的吗？

Input:你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。

Assistant: 
郑振铎_人名
阿英_人名
````

原始模型也并没有退化。

## 报错解决

- 安装mpi4py报错

```python
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev
pip install mpi4py
```

# 补充

- **怎么训练自己的数据？**
	按照instruct_data下的数据结构构造数据，定义好相关参数运行即可。
- **怎么进行预测？**
	在test.py中，预测时可根据自己的任务进行解码。
- **为什么不进行评价指标的计算？**
	只是作了初步的训练，难免效果不太好就不进行评价指标的计算了，可以在test.py里面自行定义。

# 参考

> [liucongg/ChatGLM-Finetuning: 基于ChatGLM-6B模型，进行下游具体任务微调，涉及Freeze、Lora、P-tuning等 (github.com)](https://github.com/liucongg/ChatGLM-Finetuning)
>
> [THUDM/ChatGLM-6B: ChatGLM-6B: An Open Bilingual Dialogue Language Model | 开源双语对话语言模型 (github.com)](https://github.com/THUDM/ChatGLM-6B/projects?query=is%3Aopen)
>
> [huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. (github.com)](https://github.com/huggingface/peft)
>
> https://github.com/LianjiaTech/BELLE/
