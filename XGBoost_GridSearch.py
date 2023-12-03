import numpy as np
import optuna
import pandas as pd
import random
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor

# data load
train_df=pd.read_csv('https://raw.githubusercontent.com/JeonJaeeun/machineLearning/main/house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('https://raw.githubusercontent.com/JeonJaeeun/machineLearning/main/house-prices-advanced-regression-techniques/test.csv')

all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

folds = 3
kf = KFold(n_splits=folds)

train_X = train_df_le.drop(["SalePrice", "Id"], axis=1)
train_Y = train_df_le["SalePrice"]

# SalePrice에 로그 적용
np.log(train_df['SalePrice'])

pd.options.mode.chained_assignment = None
train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]

all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)
categories = all_df.columns[all_df.dtypes == "object"]
all_df.isnull().sum().sort_values(ascending=False).head(40)

# 필요없는 피처 제거 및 결측치 처리
all_df.PoolQC.value_counts()
all_df.loc[~all_df["PoolQC"].isnull(), "PoolQC"] = 1
all_df.loc[all_df["PoolQC"].isnull(), "PoolQC"] = 0
all_df.PoolQC.value_counts()
all_df.loc[~all_df["MiscFeature"].isnull(), "MiscFeature"] = 1
all_df.loc[all_df["MiscFeature"].isnull(), "MiscFeature"] = 0
all_df.loc[~all_df["Alley"].isnull(), "Alley"] = 1
all_df.loc[all_df["Alley"].isnull(), "Alley"] = 0
HighFacility_col = ["PoolQC","MiscFeature","Alley"]
for col in HighFacility_col:
    if all_df[col].dtype == "object":
        if len(all_df[all_df[col].isnull()]) > 0:
            all_df.loc[~all_df[col].isnull(), col] = 1
            all_df.loc[all_df[col].isnull(), col] = 0
all_df["hasHighFacility"] = all_df["PoolQC"] + all_df["MiscFeature"] + all_df["Alley"]
all_df["hasHighFacility"] = all_df["hasHighFacility"].astype(int)
all_df["hasHighFacility"].value_counts()
all_df = all_df.drop(["PoolQC","MiscFeature","Alley"],axis=1)

train_df_num = train_df.select_dtypes(include=[np.number])
nonratio_features = ["Id", "MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]
num_features = sorted(list(set(train_df_num) - set(nonratio_features)))

train_df_num_rs = train_df_num[num_features]

all_df = all_df[(all_df['BsmtFinSF1'] < 2000) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['TotalBsmtSF'] < 3000) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['GrLivArea'] < 4500) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['1stFlrSF'] < 2500) | (all_df['SalePrice'].isnull())]
all_df = all_df[(all_df['LotArea'] < 100000) | (all_df['SalePrice'].isnull())]

# 숫자형 데이터를 카테고리형으로 변환
categories = categories.drop(["PoolQC","MiscFeature","Alley"])
for cat in categories:
    le = LabelEncoder()

    all_df[cat].fillna("missing", inplace=True)
    le = le.fit(all_df[cat])
    all_df[cat] = le.transform(all_df[cat])
    all_df[cat] = all_df[cat].astype("category")

train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log", "Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]

all_df["Age"] = all_df["YrSold"] - all_df["YearBuilt"]
train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]

all_df["TotalSF"] = all_df["TotalBsmtSF"] + all_df["1stFlrSF"] + all_df["2ndFlrSF"]
all_df["Total_Bathrooms"] = all_df["FullBath"] + all_df["HalfBath"] + all_df["BsmtFullBath"] + all_df["BsmtHalfBath"]
all_df["Total_PorchSF"] = all_df["WoodDeckSF"] + all_df["OpenPorchSF"] + all_df["EnclosedPorch"] + all_df["3SsnPorch"] + all_df["ScreenPorch"]
all_df["hasPorch"] = all_df["Total_PorchSF"].apply(lambda x: 1 if x > 0 else 0)
all_df = all_df.drop("Total_PorchSF",axis=1)

train_df_le = all_df[~all_df["SalePrice"].isnull()]
test_df_le = all_df[all_df["SalePrice"].isnull()]

train_df_le["SalePrice_log"] = np.log(train_df_le["SalePrice"])
train_X = train_df_le.drop(["SalePrice","SalePrice_log","Id"], axis=1)
train_Y = train_df_le["SalePrice_log"]

test_X = test_df_le.drop(["SalePrice", "Id"], axis=1)

categories = train_X.columns[train_X.dtypes == "category"]
for col in categories:
    train_X[col] = train_X[col].astype("int8")
    test_X[col] = test_X[col].astype("int8")
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2, random_state=1234, shuffle=False,  stratify=None)


# GridSearchCV를 이용한 하이퍼파라미터 튜닝 수행
param_grid = {
    "max_depth": [3, 6, 9, 12, 15],
    "colsample_bytree": [0.2, 0.4, 0.6, 0.8],
    "subsample": [0.2, 0.4, 0.6, 0.8]
}

xgb_model = XGBRegressor(learning_rate=0.05, seed=1234)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(train_X, train_Y)

best_params = grid_search.best_params_
print("GridSearchCV를 통해 찾은 최적의 하이퍼파라미터: ", best_params)

# 최적의 하이퍼파라미터를 적용하여 XGBoost 모델 재학습 및 예측
best_xgb_model = XGBRegressor(learning_rate=0.05, seed=1234, **best_params)

rmses = []
r2s = []
training_times = []

# KFold를 통한 교차 검증 (K-Fold Cross Validation) 수행
for train_index, val_index in kf.split(train_X):
    start_time = time.time()  # 학습 시작 시간 기록
    X_train, X_valid = train_X.iloc[train_index], train_X.iloc[val_index]
    y_train, y_valid = train_Y.iloc[train_index], train_Y.iloc[val_index]

    best_xgb_model.fit(X_train, y_train)
    end_time = time.time()  # 학습 종료 시간 기록
    training_time = end_time - start_time  # 학습 시간 계산
    training_times.append(training_time)  # 학습 시간 리스트에 추가

    y_pred = best_xgb_model.predict(X_valid)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    tmp_r2 = r2_score(y_valid, y_pred)

    rmses.append(tmp_rmse)
    r2s.append(tmp_r2)

# 교차 검증 결과에 대한 평균 RMSE, R^2 및 평균 학습 시간 계산 및 출력
avg_rmse = sum(rmses) / len(rmses)
avg_r2 = sum(r2s) / len(r2s)
avg_training_time = sum(training_times) / len(training_times)

print("평균 RMSE: ", avg_rmse)
print("평균 R^2: ", avg_r2)
print("평균 학습 시간: ", avg_training_time, "초")