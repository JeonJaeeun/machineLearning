import matplotlib.pyplot as plt

# 각 최적화 방법에서 얻은 평균 RMSE와 평균 R^2 데이터
mean_rmse = {
    'Grid Search': 0.12329720118005444,
    'Random Search': 0.1528936493740888,
    'Hyperopt': 0.11730916912902893,
    'Optuna': 0.11666529902612889
}

mean_r2 = {
    'Grid Search': 0.9037365468913134,
    'Random Search': 0.84831113181434,
    'Hyperopt': 0.9551129130993332,
    'Optuna': 0.9107415055603572
}

# 각 최적화 방법의 성능을 고려한 종합 평가
weights = {'RMSE': 0.5, 'R^2': 0.5}  # 가중치 설정
combined_score = {}

for method in mean_rmse.keys():
    # 각 최적화 방법의 RMSE와 R^2 값에 대해 가중 평균 계산
    weighted_score = (weights['RMSE'] * mean_rmse[method]) + (weights['R^2'] * mean_r2[method])
    combined_score[method] = weighted_score

# 가장 높은 종합 평가를 가진 최적화 방법 찾기
best_method = max(combined_score, key=combined_score.get)
best_score = combined_score[best_method]

# 첫 번째 그래프: 각 최적화 방법의 평균 RMSE와 평균 R^2 비교
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # 첫 번째 subplot (평균 RMSE)
plt.bar(mean_rmse.keys(), mean_rmse.values())
plt.title('Comparison of Mean RMSE')
plt.xlabel('Optimization Methods')
plt.ylabel('Mean RMSE')
plt.ylim(0.1, 0.16)  # y축 범위 설정
plt.yticks([0.1, 0.12, 0.14, 0.16])  # y축 눈금 설정

plt.subplot(1, 2, 2)  # 두 번째 subplot (평균 R^2)
plt.bar(mean_r2.keys(), mean_r2.values())
plt.title('Comparison of Mean R^2')
plt.xlabel('Optimization Methods')
plt.ylabel('Mean R^2')
plt.ylim(0.8, 1.0)  # y축 범위 설정
plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0])  # y축 눈금 설정

plt.tight_layout()
plt.show()

# 세 번째 그래프: 종합 평가 점수에 따른 최적화 방법 비교
plt.figure(figsize=(8, 6))
methods = list(combined_score.keys())
scores = list(combined_score.values())

# 막대 그래프 생성
plt.bar(methods, scores, color='skyblue')
plt.title('Combined Evaluation Scores of Optimization Methods')
plt.xlabel('Optimization Methods')
plt.ylabel('Combined Score')
plt.ylim(min(scores) - 0.05, max(scores) + 0.05)  # y축 범위 설정

# 가장 높은 종합 평가를 가진 최적화 방법에 표시
plt.text(methods.index(best_method), best_score, f"Best: {best_method}\nScore: {best_score:.4f}",
         ha='center', va='bottom')

# 그래프 표시
plt.tight_layout()
plt.show()
