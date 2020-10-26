# Tianchi2020ChineseMedicineQuestionGeneration
2020 阿里云天池大数据竞赛-中医药文献问题生成挑战赛

官网链接: https://tianchi.aliyun.com/competition/entrance/531826/introduction

`初赛成绩`: 0.6133(11/868)  `复赛成绩`: 0.6215(8/868)

**均为single model**

包含数据集的完整项目文件百度盘链接: `https://pan.baidu.com/s/18ZXfIU8om1EhLqttcGkEtA`  提取码：`qagl`

模型整体思路: 预训练语言模型(RoBERTa_wwm_ext_large)作为编码器, Transformer-XL作为解码器(train from scratch)，使用其他阅读理解数据集进行预学习，再在比赛数据集上进行微调

整体流程:
> 1. 数据预处理：python preprocess.py生成multi-task.pkl
> 2. 在DuReader数据集上粗粒度的预学习nohup python -u MultiTaskXLIR-DuReader train gpu-0 &  (自行设置batch-size和gpu数量)
> 3. 在DRCD和CMRC2018数据集上细粒度的预学习nohup python -u MultiTaskXLIR-DRMC train gpu-0 &
> 4. 在比赛数据集上进行学习nohup python -u MultiTaskXLIR-Final train gpu-0 final &
> 5. 使用beam_search生成测试集结果python MultiTaskXLIR-Final test gpu-0
