import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

"""
-2019-05-05,随机森林，RMSE（均方误差），0.17402
    存在问题，缺失值处理，直接drop不合理，待优化

"""


# 分析参数相关性
def analize_data():
    # 房价数据,柱形图
    print(df_train['SalePrice'].describe())
    sns.distplot(df_train['SalePrice'])
    plt.show()
    # 偏态，峰度
    print("Skewness: %f" % df_train['SalePrice'].skew())
    print("Kurtosis: %f" % df_train['SalePrice'].kurt())

    # 总平方英尺，散点图
    var = 'GrLivArea'
    # axis： 需要合并链接的轴，0是行，1是列
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.show()

    # 地下室平方英尺
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    plt.show()

    # 整体材料,取值范围1到10，越大越好，箱线图
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.show()

    # 建造年份
    var = 'YearBuilt'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ag = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    plt.show()

    # 相关系数，热点图
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()

    # 相关系数矩阵,获取相关性最高的前10个特征
    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

    # 主要特征
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size=2.5)
    plt.show()


# 缺失值分析
def missing(df_train):
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))

    df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    print(df_train.isnull().sum().max())

    return df_train


# 离群点检测，观察GrLivArea的散点图，最右侧2个点明显不符
def outliers(df_train):
    df_train.sort_values(by='GrLivArea', ascending=False)[:2]
    print("drop 2 outliers")
    print(df_train.sort_values(by='GrLivArea', ascending=False)[:2])

    df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index)
    # df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    # df_train = df_train.drop(df_train[df_train['Id'] == 523].index)

    return df_train


# 修正正态分布,原本不是线性关系，log函数处理后线性相关
def log_param():
    sns.distplot(df_train['SalePrice'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    plt.show()

    df_train['SalePrice'] = np.log(df_train['SalePrice'])
    sns.distplot(df_train['SalePrice'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    plt.show()


# 随机森林
def model(df_train):
    train_y = df_train.SalePrice
    # 根据相关度选择特征
    predictor_cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'TotRmsAbvGrd', 'YearBuilt']
    train_X = df_train[predictor_cols]

    my_model = RandomForestRegressor()
    my_model.fit(train_X, train_y)

    # GarageCars,TotalBsmtSF存在空值，
    df_test['GarageCars'] = df_test['GarageCars'].fillna(0)
    df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)

    test_X = df_test[predictor_cols]

    predicted_prices = my_model.predict(test_X)
    # 取整
    # predicted_prices = predicted_prices.astype(np.int)
    print(predicted_prices)

    my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': predicted_prices})
    my_submission.to_csv('./submission.csv', index=False)


if __name__ == "__main__":
    df_train = pd.read_csv('./input/train.csv')
    df_test = pd.read_csv('./input/test.csv')

    print(df_test.head(5))
    print(format(df_train.shape))
    print(format(df_test.shape))

    # 所有列
    # print(df_train.columns)

    # analize_data()

    # 缺失值分析，删除空值列
    df_train = missing(df_train)

    # saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
    # low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
    # high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
    # print(low_range)
    # print(high_range)

    # 离群点检测
    df_train = outliers(df_train)
    print(df_train.isnull().sum().max())

    # log_param()

    model(df_train)
