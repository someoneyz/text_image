
【天池经典打榜赛】赛道五-多模态图文检索赛解决方案

## 图文检索

#### **主要流程、对应代码**

1.模型微调（train_clip.sh）：基于电商图文数据微调Chinese-CLIP，优化中文query与商品图片的匹配能力；

2.特征提取（feature_extract.sh）：提取测试集商品图片和query的特征向量

3.跨模态检索（predict.sh）：通过计算图文特征相似度，生成Top10相关商品ID



#### 具体思路：

**①数据处理：**

训练集query一般对应1-2个商品图片，验证集和测试集query平均对应6个商品图片。训练集、验证集和测试集之间query没有交集：

**图像增强**增加训练集每个query的对应图片数量

**prompt**：

修改位置：

Chinese-CLIP/cn_clip/training/data.py  class LMDBDataset(Dataset):    def __getitem__(self, index):

Chinese-CLIP/cn_clip/eval/data.py  class EvalTxtDataset(Dataset):

```
prompts = "这是一张图片，里面存在商品："
prompt_text = prompt + raw_text
```



**②模型处理:**

- 根据官方baseline提示，使用更多的GPU和更大的batch size以取得更好的效果

- 采用lora微调：Chinese-CLIP/cn_clip/training/main2.py

- 构建不同超参数模型：Chinese-CLIP/train_clip.sh

  在验证集上测试不同超参效果，选择分数最高的微调模型进行测试集测试

  取得分数：82.81（排名 ：3） 

  对应的训练参数：

  ```
  export CUDA_VISIBLE_DEVICES=0,1
  export NCCL_DEBUG=INFO
  export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
  
  
  torchrun --nproc_per_node=2 --master_port=29500 \
      cn_clip/training/main.py \
      --train-data=/data/coding/datasets/MUGE/lmdb/train \
      --val-data=/data/coding/datasets/MUGE/lmdb/valid \
      --num-workers=4 \
      --valid-num-workers=4 \
      --resume=/data/coding/pretrained_weights/clip_cn_vit-h-14.pt \
      --reset-data-offset \
      --reset-optimizer \
      --name=muge_finetune_vit-h-14 \
      --save-step-frequency=999999 \
      --save-epoch-frequency=1 \
      --log-interval=20 \
      --report-training-batch-acc \
      --context-length=52 \
      --warmup=50 \
      --batch-size=360 \
      --valid-batch-size=360 \
      --lr=5e-06 \
      --wd=0.005 \
      --max-epochs=2 \
      --valid-step-interval=1000 \
      --valid-epoch-interval=1 \
      --vision-model=ViT-H-14 \
      --text-model=RoBERTa-wwm-ext-large-chinese\
      --use-augment \
      --grad-checkpointing
  ```

- model merge ：

  在通过不同超参数获得不同微调模型后，对模型进行融合：Chinese-CLIP\model_soup.sh

  每次将分数最高的两个模型进行此参数融合，在验证集上测试分数，若分数高于原模型集合中任意一个，替换分数最低模型，不断重复，直到融合模型分数小于模型集合中任意一个模型的分数

  模型融合过程产生的中间模型：Chinese-CLIP/logs/muge_finetune_vit-h-14/checkpoints

  模型融合过程产生的中间模型验证集测试结果：datasets/MUGE/valid_predict/valid_evaluation
 （文件过大，无法上传，可通过train_clip.sh重新训练得到）
```
epoch_latest0.pt : 79.81230031948883
epoch_latest1.pt : 82.57454739084132
epoch_latest2.pt : 83.41320553780616
epoch_latest3.pt : 83.36661341853035
merged_round1.pt : 83.68610223642173
merged_round2.pt : 83.64616613418531
merged_round3.pt : 83.57960596379127
```

最终merged_round1.pt效果最好，在测试集上取得82.97的分数（排名 ：2） 
