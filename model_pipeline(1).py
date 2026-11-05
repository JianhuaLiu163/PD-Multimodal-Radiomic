# -*- coding: utf-8 -*-
"""
完整建模流水线（与论文一致），供公开仓库使用。

流程:
  1. 读入特征矩阵 + 划分标签
  2. Z-score 标准化（训练/测试各自标准化——与论文 normalize_df(group) 一致）
  3. 训练组内: 独立样本 t 检验 (p<0.05)
  4. Pearson 相关冗余过滤 (|r|>0.9)
  5. mRMR (Rad: 30, DTL: 35 — 论文预设上限)
  6. LASSO，alpha 由 10 折 CV 确定 (random_state=0)，保留非零系数特征
  7. SMOTE (random_state=0) + 线性 SVM (Rad: C=0.005; DTL: C=0.01; random_state=0)
  8. 决策级融合: ExtraTrees(n_estimators=75, max_depth=3, random_state=0)
     以两个 SVM 的预测概率为输入，仅在训练集拟合
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ---------------- 论文固定参数（随机种子） ----------------
SPLIT_RANDOM_STATE = 4       # train_test_split 7:3
LASSO_RANDOM_STATE = 0      # LassoCV 折划分
SMOTE_RANDOM_STATE = 0      # SMOTE
SVM_RANDOM_STATE = 0        # SVM 概率估计
FUSION_RANDOM_STATE = 0     # ExtraTrees 融合器

MRMR_NUM = {'Rad': 30, 'DTL': 35}        # 进入 LASSO 的候选特征上限（论文预设）
SVM_C = {'Rad': 0.005, 'DTL': 0.01}      # SVM C（训练集上网格搜索确定）
SVM_MAX_ITER = 100


def znorm_by_group(df, featcols, group_col='group'):
    """训练/测试各组内 z-score 标准化"""
    m = df.copy()
    for c in featcols:
        for g in m[group_col].unique():
            idx = m[group_col] == g
            mu, sd = m.loc[idx, c].mean(), m.loc[idx, c].std()
            m.loc[idx, c] = (m.loc[idx, c] - mu) / sd
    return m


def correlation_filter(train_df, cols, threshold=0.9, keep=1):
    """Pearson |r|>threshold 冗余过滤。

    与论文所用 ONEKEY select_feature 算法一致：每轮对当前剩余特征（按原列序），
    统计每个特征与序在其后、|r|>threshold 的特征个数（每对只计一次），
    一次性丢弃计数 >= keep 的特征，重新计算，直至没有特征满足条件。"""
    corr = train_df[cols].corr('pearson')
    remaining = list(cols)
    while True:
        counts = {}
        for i, fi in enumerate(remaining):
            cn = 0
            for j in range(i + 1, len(remaining)):
                if abs(corr.loc[fi, remaining[j]]) > threshold:
                    cn += 1
            counts[fi] = cn
        to_drop = [f for f, n in counts.items() if n >= keep]
        if to_drop:
            for f in to_drop:
                remaining.remove(f)
        else:
            break
    return remaining


def mrmr_select(train_df, cols, num, label_col='label'):
    """mRMR 特征选择（MID）。pymrmr 不可用时退化为按 t 统计量排序取前 num 个。"""
    try:
        import pymrmr
        return pymrmr.mRMR(train_df[[label_col] + cols].astype(float), 'MID', num)
    except Exception:
        g1 = train_df[train_df[label_col] == 1][cols]
        g0 = train_df[train_df[label_col] == 0][cols]
        _, p = sps.ttest_ind(g1.values, g0.values, axis=0)
        order = np.argsort(p)
        return [cols[i] for i in order[:num]]


def lasso_select(train_df, cols, label_col='label', cv=10):
    """LASSO 10 折 CV 选 alpha，返回非零系数特征（按 |coef|>1e-6）"""
    X, y = train_df[cols], train_df[label_col]
    lc = LassoCV(alphas=np.logspace(-3, 0, 50), cv=cv, n_jobs=-1,
                 random_state=LASSO_RANDOM_STATE).fit(X, y)
    clf = Lasso(alpha=lc.alpha_).fit(X, y)
    return [f for f, c in zip(X.columns, clf.coef_) if abs(c) > 1e-6], lc.alpha_


def smote_svm(train_df, cols, C, label_col='label'):
    """SMOTE 过采样 + 线性 SVM"""
    from imblearn.over_sampling import SMOTE
    Xs, ys = SMOTE(random_state=SMOTE_RANDOM_STATE).fit_resample(train_df[cols], train_df[label_col])
    svm = SVC(probability=True, kernel='linear', max_iter=SVM_MAX_ITER,
              C=C, random_state=SVM_RANDOM_STATE).fit(Xs, ys)
    return svm


def run_pipeline(features_csv, label_csv, mrmr_num, svm_C):
    features = pd.read_csv(features_csv)
    features['ID'] = features['ID'].astype(str)
    labels = pd.read_csv(label_csv)
    labels['ID'] = labels['ID'].astype(str)
    data = features.merge(labels, on='ID')

    featcols = [c for c in features.columns if c != 'ID']
    data = znorm_by_group(data, featcols)
    train = data[data['group'] == 'train'].reset_index(drop=True)
    test = data[data['group'] != 'train'].reset_index(drop=True)

    # 1) t 检验
    g1 = train[train['label'] == 1][featcols]
    g0 = train[train['label'] == 0][featcols]
    _, p = sps.ttest_ind(g1.values, g0.values, axis=0)
    keep = [c for c, pv in zip(featcols, p) if pv < 0.05]

    # 2) 相关过滤
    sel = correlation_filter(train, keep, threshold=0.9)

    # 3) mRMR
    selm = mrmr_select(train, sel, min(mrmr_num, len(sel)))

    # 4) LASSO
    selected, alpha = lasso_select(train, selm)

    # 5) SMOTE + SVM
    svm = smote_svm(train, selected, svm_C)
    tr_p = svm.predict_proba(train[selected])[:, 1]
    te_p = svm.predict_proba(test[selected])[:, 1]

    print(f'n: t-test {len(keep)} -> corr {len(sel)} -> mRMR {len(selm)} '
          f'-> LASSO {len(selected)} (alpha={alpha:.4f})')
    print(f'  train AUC={roc_auc_score(train["label"], tr_p):.4f}  '
          f'test AUC={roc_auc_score(test["label"], te_p):.4f}')
    return train, test, tr_p, te_p


def fuse_combined(rad_tr_p, rad_te_p, dtl_tr_p, dtl_te_p, y_tr, y_te):
    """决策级融合: ExtraTrees 在训练集拟合两模型概率"""
    Xtr = np.column_stack([rad_tr_p, dtl_tr_p])
    Xte = np.column_stack([rad_te_p, dtl_te_p])
    etc = ExtraTreesClassifier(n_estimators=75, max_depth=3, min_samples_split=2,
                                random_state=FUSION_RANDOM_STATE).fit(Xtr, y_tr)
    print(f'Combined: train AUC={roc_auc_score(y_tr, etc.predict_proba(Xtr)[:, 1]):.4f}  '
          f'test AUC={roc_auc_score(y_te, etc.predict_proba(Xte)[:, 1]):.4f}')
    return etc


_FUSION_MODEL = None


def fit_combined(rad_tr_p, dtl_tr_p, y_tr):
    """拟合决策级融合器（ExtraTrees，仅用训练集），返回融合器"""
    global _FUSION_MODEL
    Xtr = np.column_stack([rad_tr_p, dtl_tr_p])
    _FUSION_MODEL = ExtraTreesClassifier(n_estimators=75, max_depth=3, min_samples_split=2,
                                           random_state=FUSION_RANDOM_STATE).fit(Xtr, y_tr)
    return _FUSION_MODEL


def combined_probability(rad_p, dtl_p):
    """融合器对给定两组概率输出联合概率。若未拟合则先以传入数据拟合（训练场景）。"""
    if _FUSION_MODEL is None:
        raise RuntimeError('请先调用 fit_combined 拟合融合器')
    return _FUSION_MODEL.predict_proba(np.column_stack([rad_p, dtl_p]))[:, 1]


if __name__ == '__main__':
    tr_r, te_r, rp_tr, rp_te = run_pipeline(
        'data/rad_features.csv', 'data/label_split.csv', MRMR_NUM['Rad'], SVM_C['Rad'])
    tr_d, te_d, dp_tr, dp_te = run_pipeline(
        'data/rad_features_dtl.csv', 'data/label_split.csv', MRMR_NUM['DTL'], SVM_C['DTL'])
    fuse_combined(rp_tr, rp_te, dp_tr, dp_te, tr_r['label'].values, te_r['label'].values)
