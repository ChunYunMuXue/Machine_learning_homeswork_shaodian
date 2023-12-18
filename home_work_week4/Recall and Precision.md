

# Recall and Precision 

## 定义

|                    |    **Predict +**    |    **Predict -**    |
| :----------------: | :-----------------: | :-----------------: |
| **Ground Truth +** | True Positive (TP)  | False Negative (FN) |
| **Ground Truth -** | False Positive (FP) | True Negative (TN)  |

**Recall** = $\frac{TP}{TP + FN} = \frac{TP}{GTP}$ = **TPR** (召回率：原有的对的实例中判断正确率) 召回率高意味可以找到图片中更多的物品

**FNR** = 1 - Recall(漏报率)

**FPR** = $\frac{FP}{FP + TN}$ （误报率）

**Precision ** = $\frac{TP}{TP + FP} = \frac{TP}{PreP}$ (精度：判断对的实例的准确率)

## 组合

### Cost

类似于地震的事件检测系统经常使用 **FNR** 和 **FPR** 的加权线性组合

在地震的例子中：

FNR 的权重 ：地震了而没有预测出来，这个代价就不仅仅是经济上的了，还可能会有人员伤亡等。

FPR 的权重 : 预测地震而实际没有的话，这个代价可能是为了预防地震而采取的行动，人员疏散以及随之带来的经济损失

**Cost** = $C_{\text{FP}} \times \text{FPR}\ \ + C_{\text{FN}} \times \text{FNR}$

### F-Score

在信息检索中经常用到 F-Score,是 Recall 和 Precision 的调和平均

$\text{F}_{\beta} = \frac{(\beta ^ 2 + 1)\text{P R}}{\beta^2 \text{P} + \text{R}}$

其中的特例 $\text{F}_1 = \frac{2\text{P R}}{\text{P} + \text{R}}$

### ROC 曲线

ROC 曲线为我们取不同阈值时的不同二分类结果的 (TPR,FPR) 的若干点对的曲线图

![figure 4](D:\article and  study\study\Dian_serach_about\machine_learning_Dian\home_work_week4\Recall and Precision.assets\12872_2021_2086_Fig4_HTML-17027180162376.png)

### AUC 

AUC 是分类的重要评估指标，**AUC值越大预测准确率越高** 。

AUC（Area Under Curve）被定义为ROC曲线下与坐标轴围成的面积。![img](D:\article and  study\study\Dian_serach_about\machine_learning_Dian\home_work_week4\Recall and Precision.assets\v2-8743fe973c693233786936b9add55512_720w-17027180144825.webp)

### PR 曲线

和 ROC 曲线同理，我们使用 (Recall,Precision) 点对画图得到 PR 曲线![img](D:\article and  study\study\Dian_serach_about\machine_learning_Dian\home_work_week4\Recall and Precision.assets\20180521222743741-17027180119364.jpeg)

### AP

AP 同 AUC 意义，就是曲线下面积

$\text{AP} = \int_{0}^1\text{P}(R)\,dR$

我猜可以使用辛普森积分法 ？

### 人脸识别中的组合的想法

我认为在人脸识别中 **Preceision** 是最重要的指标，对于 **Recall** 如果召回率较低的缺陷可以通过多次检测来一定程度上弥补，但如果 **Precision** 过低，把会造成数据，经济等的严重的泄露损失

我的想法是应该在 PR 图像上找到 **Precision** 的最靠右的最大值点

----------------------------------------------------------------------------

### 任务5

对于目标检测任务来说，如果我们不关注边界框的proposal，模型的输入、输出、训练会有什么问题，有哪些弊端？

我认为如果不关注边界框的proposal,模型的输出就会变为所有输入的边界框统计意义上的分布，最终很有可能只得到趋于一个几何意义上的平衡点，导致最终模型失去效力。

