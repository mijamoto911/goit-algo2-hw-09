import matplotlib.pyplot as plt
import numpy as np


# Визначення функції Сфери
def sphere_function(x):
    return sum(xi**2 for xi in x)


# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    dim = len(bounds)
    lower_bound, upper_bound = bounds
    current_x = np.random.uniform(lower_bound, upper_bound, dim)
    current_value = func(current_x)

    best_x, best_value = current_x, current_value
    history = []

    for _ in range(iterations):
        history.append(current_value)

        neighbor_x = current_x + np.random.uniform(-0.1, 0.1, dim)
        neighbor_x = np.clip(neighbor_x, lower_bound, upper_bound)

        neighbor_value = func(neighbor_x)

        if neighbor_value < current_value:
            current_x, current_value = neighbor_x, neighbor_value

        if current_value < best_value:
            best_x, best_value = current_x, current_value

        if abs(best_value - current_value) < epsilon:
            break

    return best_x.tolist(), best_value, history


# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    dim = len(bounds)
    lower_bound, upper_bound = bounds
    best_x = np.random.uniform(lower_bound, upper_bound, dim)
    best_value = func(best_x)
    history = []

    for _ in range(iterations):
        history.append(best_value)
        candidate_x = np.random.uniform(lower_bound, upper_bound, dim)
        candidate_value = func(candidate_x)

        if candidate_value < best_value:
            best_x, best_value = candidate_x, candidate_value

        if abs(best_value - candidate_value) < epsilon:
            break

    return best_x.tolist(), best_value, history


# Simulated Annealing
def simulated_annealing(
    func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6
):
    dim = len(bounds)
    lower_bound, upper_bound = bounds
    current_x = np.random.uniform(lower_bound, upper_bound, dim)
    current_value = func(current_x)

    best_x, best_value = current_x, current_value
    history = []

    for _ in range(iterations):
        history.append(current_value)

        neighbor_x = current_x + np.random.uniform(-0.1, 0.1, dim)
        neighbor_x = np.clip(neighbor_x, lower_bound, upper_bound)

        neighbor_value = func(neighbor_x)
        delta = current_value - neighbor_value
        acceptance_probability = np.exp(delta / (temp + 1e-10))

        if neighbor_value < current_value or np.random.rand() < acceptance_probability:
            current_x, current_value = neighbor_x, neighbor_value

        if current_value < best_value:
            best_x, best_value = current_x, current_value

        if abs(best_value - current_value) < epsilon or temp < epsilon:
            break

        temp *= cooling_rate

    return best_x.tolist(), best_value, history


if __name__ == "__main__":
    # Межі для функції
    bounds = (-5, 5)

    print("Hill Climbing:")
    hc_solution, hc_value, hc_history = hill_climbing(sphere_function, bounds)
    print("Розв'язок:", hc_solution, "Значення:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value, rls_history = random_local_search(sphere_function, bounds)
    print("Розв'язок:", rls_solution, "Значення:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value, sa_history = simulated_annealing(sphere_function, bounds)
    print("Розв'язок:", sa_solution, "Значення:", sa_value)


""" Алгоритм	        Швидкість	     Глобальна оптимальність	Коментар
 Hill Climbing	        ✅ Швидкий	    ❌ Локальний мінімум	Застрягає в локальних мінімумах
 Random Local Search	❌ Повільний	    ❌ Локальний мінімум	Більший пошук, але без стратегії
 Simulated Annealing	✅ Добрий баланс	✅ Краще глобальне рішення	Ефективний, не застрягає """
