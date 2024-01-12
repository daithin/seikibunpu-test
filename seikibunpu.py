import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_normal_data(mean, std_dev, size):
    data = np.random.normal(loc=mean, scale=std_dev, size=size)
    return data

# 正規分布のパラメータを指定
true_mean = 50
true_std_dev = 10
data_size = 1000  # 生成するデータの数

# 正規分布に従うデータを生成
generated_data = generate_normal_data(true_mean, true_std_dev, data_size)
print(generated_data)


# 生成されたデータのヒストグラムを描画
plt.hist(generated_data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='生成されたデータ')

# 最尤推定を使用して正規分布をフィット
a=int(len(generated_data))
kazu=int(input())
b=int(a/kazu)
mle_mean, mle_std_dev = norm.fit(generated_data[:b])

# フィットされた正規分布の確率密度関数を描画
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mle_mean, mle_std_dev)
plt.plot(x, p, 'r', linewidth=2, label='MLEによる正規分布')

# 真の正規分布の確率密度関数を描画
p_true = norm.pdf(x, true_mean, true_std_dev)
plt.plot(x, p_true, 'k', linewidth=2, label='真の正規分布')

# グリッド線を追加
plt.grid(True, linestyle='--', alpha=0.7)

# 凡例を表示
plt.legend()

# グラフのラベルとタイトルを設定
plt.title('Maximum Likelihood Estimation (MLE) for Normal Distribution with Reduced Data')
plt.xlabel('Value')
plt.ylabel('Probability Density')

# グラフを表示
plt.show()

# 真の正規分布のパラメータを表示
print("真の正規分布のパラメータ:")
print("平均:", true_mean)
print("標準偏差:", true_std_dev)

# MLEによって推定された正規分布のパラメータを表示
print("\nMLEによる正規分布の推定パラメータ:")
print("平均:", mle_mean)
print("標準偏差:", mle_std_dev)
