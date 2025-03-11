# 🧠 Локальний пошук, евристики та імітація відпалу

## 📖 Опис

Цей проєкт реалізує **три методи локальної оптимізації** для мінімізації **функції Сфери**:

- 🔵 **Hill Climbing (Підйом на гору)**
- 🔴 **Random Local Search (Випадковий локальний пошук)**
- 🟢 **Simulated Annealing (Імітація відпалу)**

### 📌 **Функція Сфери**

Функція Сфери визначається як:

\[
f(x) = \sum\_{i=1}^{n} x_i^2
\]

Діапазон значень змінних: **x ∈ [-5, 5]**.

---

## 🚀 Запуск проєкту

### 🛠 Встановлення залежностей

Перед запуском переконайтеся, що у вас встановлено **Python 3** та необхідні бібліотеки:

```
pip install numpy matplotlib
```

## ▶ Запуск коду

```
python sphere_function.py
```

### 📊 Опис алгоритмів

| `Алгоритм`            | `Швидкість`      | `Глобальна оптимальність`                           | `Коментар`                      |
| --------------------- | ---------------- | --------------------------------------------------- | ------------------------------- |
| `Hill Climbing`       | ✅ Швидкий       | ❌ Локальний мінімум                                | Застрягає в локальних мінімумах |
| `Random Local Search` |  ❌ Повільний    | ❌ Локальний мінімум                                | Випадковий пошук без структури  |
| `Simulated Annealing` | ✅ Добрий баланс | ✅ Краще глобальне рішення Використовує ймовірність | прийняття гірших рішень         |

## 📌 Деталі реалізації

🔵 Hill Climbing

1. Починає з випадкової точки.
2. Перевіряє сусідів із невеликими змінами.
3. Якщо знайдено краще рішення – рухається до нього.
4. Якщо покращень немає – зупиняється.

❗ Проблема: може застрягти в локальному мінімумі.

🔴 Random Local Search

1. Генерує випадкові точки в межах простору.
2. Записує найкраще знайдене значення.
3. Не використовує попередню точку – лише випадкові перевірки.

❗ Проблема: може бути повільним, оскільки досліджує занадто багато непотрібних точок.

---

🟢 Simulated Annealing

1. Починає з випадкової точки.
2. Генерує сусіда із випадковими змінами.
3. Приймає гірші рішення з певною ймовірністю (охолодження).
4. Чим нижча температура, тим менша ймовірність прийняття гірших рішень.
5. Допомагає уникнути локальних мінімумів.

✅ Найкращий баланс між швидкістю та якістю рішення.

### 🔥 Приклад виводу в консолі:

````Hill Climbing:
Розв'язок: [0.00231, -0.00154] Значення: 1.4e-6

Random Local Search:
Розв'язок: [0.084, -0.072] Значення: 0.0125

Simulated Annealing:
Розв'язок: [0.00089, -0.00056] Значення: 7.2e-7 ```

````

Автор: Eduard Schumacher Email: mijamoto911@gmail.com
