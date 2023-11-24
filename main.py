import math

import numpy as np

def f(variables, n):
    if n == 4:
        x, y, z, t = variables
        return np.array(
            [x + 2 * y + z + 4 * t - 20.700, x ** 2 + 2 * x * y + t ** 3 - 15.880, x ** 3 + z ** 2 + t - 21.218, 3 * y + z * t - 7.900])
    elif n == 3:
        x, y, z = variables
        return np.array([x ** 2 - y - 1, x + y ** 2 - z - 4, x - z ** 2 + y * z - 2])
    elif n == 2:
        x, y = variables
        return np.array([np.exp(x) - np.exp(y)-1, x ** 3 + y ** 3 - 1])


def calculate_jacobian_matrix(variables, n):
    h = 0.0001
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            delta = np.zeros_like(variables)
            delta[j] = h
            df_dx = (f(variables + delta, n)[i] - f(variables, n)[i]) / h
            row.append(df_dx)
        result.append(row)
    return np.array(result)


def newton_method(initial_values, n, tolerance=0.0001, max_iterations=10000):
    variables = np.array(initial_values, dtype=float)

    for k in range(max_iterations):
        system_eq = f(variables, n)
        jacobian = calculate_jacobian_matrix(variables, n)

        rhs = -system_eq
        try:
            delta_variables = np.linalg.solve(jacobian, rhs)
        except np.linalg.LinAlgError:
            print("Сингулярна матриця Якобі. Метод Ньютона не може продовжуватися.")
            return

        variables += delta_variables

        print(f"Ітерація {k + 1}: Змінні = {variables}")

        if np.all(np.abs(delta_variables) < tolerance):
            print("Умова збіжності виконалась.")
            break


if __name__ == "__main__":
    try:
        n = int(input())
        if n not in [2, 3, 4]:
            raise ValueError("Неправильні дані. Будь ласка, введіть 2, 3 або 4.")

        initial_values = [float(input(f"Введіть початкове значення для x{i + 1}: ")) for i in range(n)]

        newton_method(initial_values, n)
    except ValueError as e:
        print(f"Error: {e}")
