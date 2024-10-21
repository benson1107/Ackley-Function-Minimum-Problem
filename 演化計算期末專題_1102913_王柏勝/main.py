import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga #沒用上，演化過程不使用套件
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D    #繪製3D圖

# ackley function 定義
def ackley_function(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)

# 演化算法frame定義
def evolutionary_algorithm(pop_size, num_generations, mutation_rate, mutation_range):
    # 初始化個體
    population = np.random.rand(pop_size, 2) * 10 - 5  # 生成[-5, 5]內的random實數

    # 記錄每代最佳解和適應值及輪盤法個體選擇機率
    best_solutions = []
    best_fitness_values = []
    selection_probabilities = []

    # 生成x和y座標
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, ackley_function(x, y), cmap='viridis', alpha=0.5)
    """

    for generation in range(num_generations):
        # 評估個體適應度
        fitness = np.apply_along_axis(lambda ind: ackley_function(ind[0], ind[1]), 1, population)

        # 找到最佳個體
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # 紀錄每一代最佳個體和適應值，以及當前輪盤法機率
        best_solutions.append(best_individual)
        best_fitness_values.append(fitness[best_idx])
        selection_probs = fitness / np.sum(fitness)
        selection_probabilities.append(selection_probs)

        # 輸出目前本代最佳適應值和個體
        print(f"Generation {generation + 1}: Best Fitness = {fitness[best_idx]}, Best Individual = {best_individual}")

        """
        # 輸出輪盤法的個體間選擇機率  (為避免output過於冗長，可先拿掉)
        print(f"Selection Probabilities: {selection_probs}")
        """

        # 輪盤法(Roulette Wheel) selection
        selected_indices = np.random.choice(pop_size, pop_size, p=selection_probs)
        selected_population = population[selected_indices]
        # 選擇精英個體
        elite_idx = np.argmin(fitness)
        elites = population[elite_idx]

        # 生成新一代個體
        offspring = np.random.rand(pop_size, 2) * 10-5 # 隨機生成新個體
        offspring[:1] = elites  # 以elitism保留個體

        # mutation
        mutation = np.random.normal(0, mutation_range, size=(pop_size, 2))
        mutation_mask = np.random.rand(pop_size, 2) < mutation_rate
        offspring += mutation * mutation_mask

        # 族群update
        population = offspring
        """
        # 繪製演化算法的點
        ax.scatter(population[:, 0], population[:, 1], fitness, label=f'Generation {generation + 1}')





    # 設置標題和軸標籤
    ax.set_title('Ackley Function and Evolutionary Algorithm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 顯示圖形
    plt.legend()
    plt.show()
    """
    return best_solutions, best_fitness_values, selection_probabilities

# 參數設定
pop_size = 20
num_generations = 500 # 代數較少以減少圖表數量，可根據需要增加
mutation_rate = 0.1
mutation_range = 0.2

# 執行演化算法
best_solutions, best_fitness_values, selection_probabilities = evolutionary_algorithm(pop_size, num_generations, mutation_rate, mutation_range)

# 畫出演化過程二維圖表
generations = list(range(1, num_generations + 1))
plt.plot(generations, best_fitness_values, label='Best Fitness')
plt.title('Evolutionary Algorithm Evolution Process for Ackley Function')
plt.xlabel('Generations')
plt.ylabel('Fitness Value (Ackley Function)')
plt.legend()
plt.grid(True)
plt.show()

# print result
final_best_solution = best_solutions[-1]
final_best_fitness = best_fitness_values[-1]
print(f"\nFinal Best Solution: {final_best_solution}, Minimum Value: {ackley_function(final_best_solution[0], final_best_solution[1])}")
