import numpy as np
import optuna
import pandas as pd
import random
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

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

# XGBoost 모델 최적화를 위한 Objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 공간 정의
    xgb_params = {
    "learning_rate":0.05,
    "seed":1234,
    "max_depth":trial.suggest_int("max_depth",3,16),
    "colsample_bytree":trial.suggest_uniform("colsample_bytree",0.2,0.9),
    "subsample":trial.suggest_uniform("subsample",0.2,0.9),
    }
    
    # 데이터를 DMatrix 형태로 변환
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]

    # XGBoost 모델 학습
    model_xgb = xgb.train(xgb_params, xgb_train,
                            evals=evals,
                            num_boost_round=1000,
                            early_stopping_rounds=20,
                            verbose_eval=10,)
    
    # 검증 데이터에 대한 예측값 생성
    y_pred = model_xgb.predict(xgb_eval)
    # RMSE 계산
    score = np.sqrt(mean_squared_error(y_valid, y_pred))
    return score

# Optuna를 사용한 하이퍼파라미터 튜닝 수행
# Optuna의 RandomSampler를 이용해 시드 값을 0으로 설정하여 샘플러 생성
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
# Objective 함수를 최대 10번(trials) 실행하여 최적의 하이퍼파라미터 탐색
study.optimize(objective, n_trials=10)
# 가장 무수한 성능의 하이퍼파라미터 출력
#study.best_params

# Optuna를 통해 찾은 최적의 하이퍼파라미터 설정
best_params = study.best_params

# Optuna에서 찾은 최적의 하이퍼파라미터로 설정
xgb_params = {
    "learning_rate": 0.05,
    "seed": 1234,
    "max_depth": best_params["max_depth"],
    "colsample_bytree": best_params["colsample_bytree"],
    "subsample": best_params["subsample"]
}

models_xgb = []
rmses_xgb = []
oof_xgb = np.zeros(len(train_X))
training_times = []  # 모델 학습 시간을 저장할 리스트

r2_scores = []

# KfOLD를 통한 교차 검증 (K-Fold Cross Validation) 수행
for train_index, val_index in kf.split(train_X):
    start_time = time.time()  # 학습 시작 시간 기록
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]

    # XGBoost 모델 학습
    model_xgb = xgb.train(xgb_params, xgb_train,
                          evals=evals,
                          num_boost_round=1000,
                          early_stopping_rounds=20,
                          verbose_eval=20,)
    end_time = time.time()  # 학습 종료 시간 기록
    training_time = end_time - start_time  # 학습 시간 계산
    training_times.append(training_time)  # 학습 시간 리스트에 추가

    y_pred = model_xgb.predict(xgb_eval)
    tmp_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    models_xgb.append(model_xgb)
    rmses_xgb.append(tmp_rmse)
    oof_xgb[val_index] = y_pred

    r2 = r2_score(y_valid, y_pred)
    r2_scores.append(r2)

# Optuna를 통해 찾은 최적의 하이퍼파라미터 출력
print("Optuna를 통해 찾은 최적의 하이퍼파라미터: ", best_params)

# 교차 검증 결과에 대한 평균 RMSE 계산 및 출력
avg_rmse = sum(rmses_xgb) / len(rmses_xgb)
print("평균 RMSE: ", avg_rmse)

# 평균 R^2 출력
avg_r2 = np.mean(r2_scores)
print("평균 R^2: ", avg_r2)

# 전체 모델이 평균 학습 시간 계산 및 출력
avg_training_time = sum(training_times) / len(training_times)
print("평균 학습 시간: ", avg_training_time, "초")

