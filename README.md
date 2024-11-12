# README

## 概览
项目：Porto Seguro’s Safe Driver Prediction
目的：预测汽车保险单持有人提出索赔的概率
本质：分类问题，输出结果是一个bool值（yes or no）
指标：使用Normalized Gini Coefficient(归一化基尼系数进行评估)
数据集细节：
(1)测试集的数据量超过训练集
(2)匿名数据，没有用户信息，只有id、target、features


## 数据集纯度指标
在决策树的学习过程中，算法会利用纯度指标来选择最佳的分裂特征和分裂点，以便构建出纯度尽可能高的节点
通过不断地分裂和纯化节点，最终可以形成一个完整的决策树模型，用于对新样本进行分类或回归预测
常见的纯度指标有：熵、Gini系数等等

### 信息熵(entropy)和信息增益
**定义**
熵通过计算数据集中每个类别出现的概率的对数的负加权和来量化纯度
**公式**
H(p) = - sum((p_i)*log2(p_i)) , i from 1 to n
其中p_i为第i类样本在数据集；熵的值越小，数据集的纯度越高


### 基尼指数(Gini index)
**定义**
Gini指数通过计算数据集中**随机选择**两个样本它们类别不同的概率来量化纯度；在决策树算法中，Gini指数被用作选择分割点和评估决策树质量的一种方法
**计算公式**
Gini(p) = 1 - sum((p_i)^2) , i from 1 to n
其中p_i为第i类样本在数据集中的比例；Gini指数的值越小，数据集的纯度越高(节点纯度高指的是节点包含的样本尽量属于同一类别)
**作用**
(1)在决策树中，Gini指数被用作分裂节点选择的标准；对于某个特征的不同分割点，计算使用这个分割点分裂后的两个子节点的加权Gini指数之和，然后取最小值，这个最小值对应的分割点就是选择的分裂节点(相当于最小化每个分割节点的不纯度)
(2)剪枝是防止决策树过拟合的常用技术，在剪枝的过程中，Gini指数可以帮助评价剪枝后决策树的性能；如果剪枝后的决策树能够保持较低的Gini指数，表明剪枝没有显著影响模型的预测能力



### 基尼系数(Gini Coefficient)





## 数据预处理





### 特征选择

学习参考：

https://www.kaggle.com/code/ogrellier/noise-analysis-of-porto-seguro-s-features/comments

Elements of Statistical Learning 2ed  p593

https://anotherdataminingblog.blogspot.com/2013/10/techniques-to-improve-accuracy-of-your_17.html 视频的40:12分左右开始讲variable importance部分



#### 衡量变量重要性(variable importance)的算法

目的：用于提取重要特征

优点：(1)可以消除冗余变量；(2)这个算法是独立于模型的，换句话说，对各类不同模型都可以使用这个方法

做法：重要看"变化"，将所有变量通过模型，观察产生的Error = Predict - Actual，Error增加越多说明变量越重要(能对模型产生很大的影响)，若Error没有变化说明它不怎么重要



#### 为什么对特征进行这样的组合构成新特征？





#### 为什么要对组合特征使用LabelEncoder？





#### 要不要做数据增强？

- **加噪**

  ```
  # 给特征数据添加噪声，防止过拟合
  def add_noise(series, noise_level):
      return series * (1 + noise_level * np.random.randn(len(series)))
  ```

  这是一个可插拔模块，它要不要使用取决于在训练集上做交叉验证后模型的泛化能力是否得到提升/是否存在过拟合问题，可以控制变量，对比加噪和未加噪时模型的效果

  对于噪声水平`noise_level`，可以通过网格搜索或超参数优化的方式来确定

  我在这里使用的是高斯噪声(也即噪声服从标准正态分布)，对于一些特定问题还可以使用均匀噪声、椒盐噪声等

- **上采样**