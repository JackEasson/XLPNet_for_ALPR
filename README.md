# XLPNet：一个基于纯卷积神经网络的轻量化车牌检测与识别算法

<div align=center>
<img src="./examples/XLPNet_600×300.png"/>
</div>
![Alt]("./examples/XLPNet_600×300.png" #pic_center)

## 1 详细介绍见CSDN
[XLPNet车牌识别算法详解系列文章](https://blog.csdn.net/qq_42891019/category_11691340.html?spm=1001.2014.3001.5482)
欢迎点赞收藏！

## 2 项目简介
- 一个诞生于2022年，基于纯卷积神经网络结构的**车牌检测与识别系统**
- XLPNet分为车牌检测与车牌字符识别两部分，具备**轻量化**、**运行高效**的特点
- 算法上具备一些新颖的设计与思考，并非现有方案的简单集成
- 基于**pytorch**构建完整的网络结构与训练体系，支持GPU/多GPU训练
- 提供完整的**opencv调用程序**（包括python版本和C++版本），可以完全摆脱深度学习框架的限制，便于在多种不同平台和系统（Windows/Linux/Raspbian）直接或间接部署使用

很多细节上参考了我之前的工作SLPNet：[SLPNet_pytorch](https://github.com/JackEasson/SLPNet_pytorch)

XLPNet可视为SLPNet的改进版
<div align=center>
<img src="./examples/SLPNet.JPG"/>
</div>

## 3 目前完成度
- [x] 车牌检测（pytorch模型训练与opencv调用）
- [ ] 车牌识别
- [ ] 完整系统整合

## 4 示例
<div align=center>
<img src="./examples/detection_show.jpg" width=800/>
</div>

## 5 使用
详见博客：(https://blog.csdn.net/qq_42891019/article/details/123797612)[https://blog.csdn.net/qq_42891019/article/details/123797612]
### 5.1 环境要求
pytorch >= 1.6.0<br>
opencv >= 4.2.0<br>
tqdm<br>
numpy<br>
### 5.2 使用
#### 5.2.1 训练
可以提前修改``config.py``中的内容，大部分超参数可以通过此文件修改
- CPU / 单GPU训练：<br>
``python train.py --device 'cpu' / 0 / 1/ ... --epoch xxx --batch_size xxx --savedir xxx``
- CPU / 单GPU / 多GPU并行训练：<br>
``python train_multi_gpu.py --device 'cpu' / 0 / 1/ ... --epoch xxx --batch_size xxx --savedir xxx``<br>
注意：这里多GPU训练所得的模型需要转换保存为普通模型，使用``trans_tools.py``中的
``multi_gpu2normal_model()``函数处理。
#### 5.2.2 测试
- 基于角点（高斯得分）的AP测试：``python test.py --weight xxx --savedir xxx(保存测试结果的txt所在路径)``<br>
整体流程与传统检测任务AP计算一致，但不是计算预测bbox与GT的IoU，而是**采用高斯得分来判定真正例**。这里的GT是CCPD数据标注中的真实角点信息。
- 基于一般化边界框的AP测试：``python test_ccpd.py --weight xxx --savedir xxx(保存测试结果的txt所在路径)``<br>
采用的是将检测所得角点取其外接正矩形，**与GT的边界框计算IoU以判定真正例**，计算AP。这里的GT是CCPD数据标注中的真实边界框信息。

注意，两个测试程序中都包含`split_mode`选项，其作用是是否单独对测试集中每个CCPD子集进行AP测试并输出到txt中:
- `--split_mode False` 测试集整体计算AP
- `--split_mode True` 测试集子集单独计算AP
#### 5.2.3 演示
``python demo.py --device xxx ......``
其中`weight`为训练好的pth模型地址，注意`config.py`中的`INP_SIZE`与`BACKBONE`要与模型对应一致，`img_path`为输入文件夹/图片路径，即输入是一张图还是多幅图片所在的文件夹都可以，`savedir`为输出图片保存的位置，保存名称与输入图片相同。
