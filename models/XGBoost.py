

"""
介绍:(1)使用xgboost的评估结果来获取当前fold和跨folds的最佳轮次
    (2)采用一种上采样方法限制了交叉验证中的过拟合
参考: https://www.kaggle.com/code/ogrellier/xgb-classifier-upsampling-lb-0-283

This simple scripts demonstrates the use of xgboost eval results to get the best round
for the current fold and accross folds. 
It also shows an upsampling method that limits cross-validation overfitting.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold # 用于进行分层k-folds cross-validation
import gc # 垃圾收集器，可手动调用提高运行效率
from numba import jit
from sklearn.preprocessing import LabelEncoder
import time


@jit
def eval_gini(y_true, y_prob):
    """
    计算基尼系数（Gini coefficient）的函数
    使用了基于洛伦兹曲线(Lorenz curve)的离散化计算方法，有效避免了积分的计算，是一种近似方法
    
    Args:
        y_true (list or numpy.ndarray): 真实标签，包含0和1
        y_prob (list or numpy.ndarray): 预测概率，对应于y_true中每个样本属于正类的概率
    
    Returns:
        float: 计算得到的基尼系数。
    
    参考: https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true) # 将y_true转换为numpy数组

    # 使用np.argsort(y_prob)获取y_prob排序后的索引,然后根据这些索引对y_true进行排序
    y_true = y_true[np.argsort(y_prob)]
    
    ntrue = 0 # 累计正样本数
    gini = 0
    delta = 0 # 累计负样本数(在当前样本之前的)
    n = len(y_true)
    # 逆序遍历(因为排序顺序是按照预测概率从低到高排序，因此先获取概率高的数据)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta # (y_i * delta)表示样本i对基尼系数的贡献
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    """
    给特征数据添加噪声，防止过拟合
    
    Args:
        series (numpy.ndarray): 输入的时间序列数据
        noise_level (float): 噪声水平，表示噪声相对于原始序列的比例
    
    Returns:
        numpy.ndarray: 添加噪声后的时间序列数据
    
    """

    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                tst_series=None,
                tarrget=None,
                min_samples_leaf=1,
                smoothing=1,
                noise_level=0):
    """

    trn_series: 训练集中的类别特征，类型为 Pandas Series
    tst_series: 测试集中的类别特征，类型为 Pandas Series
    target: 目标变量（可以是连续的或分类的），它用于计算每个类别的均值
    min_samples_leaf: 对每个类别考虑计算平均值的最小样本数
    smoothing: 平滑系数，用于平衡类别均值与整体均值之间的相对权重，避免对样本数很少的类别过拟合
    平滑度计算参考：https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    参考：https://www.saedsayad.com/encoding.htm、
        https://www.kaggle.com/code/ogrellier/python-target-encoding-for-categorical-features/notebook
    
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    # 通过assert检查输入的有效性,若未通过会抛出AssertionError异常
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name

    # 将特征值和目标变量拼接，计算目标变量的均值等
    temp = pd.concat([trn_series, target], axis=1)
    # 计算目标变量均值
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # 计算平滑系数
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # 计算整体的(先验)均值
    prior = target.mean()
    # 使用平滑系数调整类别平均值
    # 类别中样本数越大，目标变量整体均值在最终编码值中所占的权重就越小
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # 使用map()将训练集和测试集中的分类特征值映射到最终编码值上
    # 对于未出现在训练集中的类别（导致NaN），用整体平均值填充
    ft_trn_series = trn_series.map(averages['target']).fillna(prior)  
    ft_tst_series = tst_series.map(averages['target']).fillna(prior) 

    # ft_trn_series = pd.merge(
    #     trn_series.to_frame(trn_series.name),
    #     averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
    #     on=trn_series.name,
    #     how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # # pd.merge does not keep the index so restore it
    # ft_trn_series.index = trn_series.index
    # ft_tst_series = pd.merge(
    #     tst_series.to_frame(tst_series.name),
    #     averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
    #     on=tst_series.name,
    #     how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # # pd.merge does not keep the index so restore it
    # ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

gc.enable()

# 读取数据集
trn_df = pd.read_csv("../data/train.csv", index_col=0)
sub_df = pd.read_csv("../data/test.csv", index_col=0)

target = trn_df["target"]
del trn_df["target"]

# 这里的特征值已经是降维操作过的(33个/57个)
# 特征选择参考：https://www.kaggle.com/code/ogrellier/noise-analysis-of-porto-seguro-s-features
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
	"ps_car_11_cat" # Very nice spot from Tilii : https://www.kaggle.com/tilii7
]

# 对给定特征组合进行处理，生成新的特征
# 为什么对这些特征进行组合？
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),  
    ('ps_reg_01', 'ps_car_04_cat'),
]
start = time.time()
"""
应用label encoding来处理新的组合特征
"""
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2 # 为生成的新特征起名(将两个特征的名字相连)
    print('current feature %60s %4d in %5.1f'
        % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
    sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
    trn_df[name1] = lbl.transform(list(trn_df[name1].values))
    sub_df[name1] = lbl.transform(list(sub_df[name1].values))

    train_features.append(name1)
    
trn_df = trn_df[train_features]
sub_df = sub_df[train_features]

# 为什么单独处理含_cat的特征？
f_cats = [f for f in trn_df.columns if "_cat" in f]

for f in f_cats:
    trn_df[f + "_avg"], sub_df[f + "_avg"] = target_encode(trn_series=trn_df[f],
                                        tst_series=sub_df[f],
                                        target=target,
                                        min_samples_leaf=200,
                                        smoothing=10,
                                        noise_level=0)

n_splits = 5
n_estimators = 200
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15) 
imp_df = np.zeros((len(trn_df.columns), n_splits))
xgb_evals = np.zeros((n_estimators, n_splits))
oof = np.empty(len(trn_df))
sub_preds = np.zeros(len(sub_df))
increase = True
np.random.seed(0)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
    val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

    clf = XGBClassifier(n_estimators=n_estimators,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=.1, 
                        subsample=.8, 
                        colsample_bytree=.8,
                        gamma=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        nthread=2)
    # Upsample during cross validation to avoid having the same samples
    # in both train and validation sets
    # Validation set is not up-sampled to monitor overfitting
    if increase:
        # Get positive examples
        pos = pd.Series(trn_tgt == 1)
        # Add positive examples
        trn_dat = pd.concat([trn_dat, trn_dat.loc[pos]], axis=0)
        trn_tgt = pd.concat([trn_tgt, trn_tgt.loc[pos]], axis=0)
        # Shuffle data
        idx = np.arange(len(trn_dat))
        np.random.shuffle(idx)
        trn_dat = trn_dat.iloc[idx]
        trn_tgt = trn_tgt.iloc[idx]
        
    clf.fit(trn_dat, trn_tgt, 
            eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
            eval_metric=gini_xgb,
            early_stopping_rounds=None,
            verbose=False)
            
    # Keep feature importances
    imp_df[:, fold_] = clf.feature_importances_

    # Find best round for validation set
    xgb_evals[:, fold_] = clf.evals_result_["validation_1"]["gini"]
    # Xgboost provides best round starting from 0 so it has to be incremented
    best_round = np.argsort(xgb_evals[:, fold_])[::-1][0]

    # Predict OOF and submission probas with the best round
    oof[val_idx] = clf.predict_proba(val_dat, ntree_limit=best_round)[:, 1]
    # Update submission
    sub_preds += clf.predict_proba(sub_df, ntree_limit=best_round)[:, 1] / n_splits

    # Display results
    print("Fold %2d : %.6f @%4d / best score is %.6f @%4d"
        % (fold_ + 1,
            eval_gini(val_tgt, oof[val_idx]),
            n_estimators,
            xgb_evals[best_round, fold_],
            best_round))
        
print("Full OOF score : %.6f" % eval_gini(target, oof))

# Compute mean score and std
mean_eval = np.mean(xgb_evals, axis=1)
std_eval = np.std(xgb_evals, axis=1)
best_round = np.argsort(mean_eval)[::-1][0]

print("Best mean score : %.6f + %.6f @%4d"
    % (mean_eval[best_round], std_eval[best_round], best_round))
    
importances = sorted([(trn_df.columns[i], imp) for i, imp in enumerate(imp_df.mean(axis=1))],
                    key=lambda x: x[1])

for f, imp in importances[::-1]:
    print("%-34s : %10.4f" % (f, imp))
    
sub_df["target"] = sub_preds

sub_df[["target"]].to_csv("submission.csv", index=True, float_format="%.9f")