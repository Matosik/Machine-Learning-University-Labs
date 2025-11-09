import numpy as np
import matplotlib.pyplot as plt
import scipy

M_0 = np.array([0, 1])
M_1 = np.array([-1, -1])
M_2 = np.array([2, 1])

B_0 = np.array([[0.35, 0.15], [0.1, 0.35]])

B_1 = np.array([[0.45, 0.15], [0.15, 0.45]])
B_2 = np.array([[0.15, 0.02], [0.02, 0.15]])
B_3 = np.array([[0.25, -0.17], [-0.17, 0.25]])


def Mahalanobis_distance(M0, M1, B):
    return (M1 - M0) @ np.linalg.inv(B) @ np.transpose(M1 - M0)


def boundary_of_bayes_classifier_for_N_with_same_B(x, M_l, M_j, B, threshold):
    M_dif = M_l - M_j
    M_dif_T = M_dif.reshape(1, 2)
    M_sum = (M_l + M_j)
    M_sum_T = M_sum.reshape(1, 2)
    B_inv = np.linalg.inv(B)

    # d_lj = d_l - d_j = 0
    # d_lj = ... = ax+b=0
    # a_0*x_0 + a_1*x_1 = const
    a = np.matmul(M_dif_T, B_inv)
    b = -0.5 * np.matmul(np.matmul(M_sum_T, B_inv), M_dif) + threshold  # page 31 случай 2

    return np.array((-b - a[0, 0] * x) / a[0, 1])


def get_count_fail(x, M_l, M_j, B, threshold):
    M_dif = M_l - M_j
    M_dif_T = M_dif.reshape(1, 2)
    M_sum = (M_l + M_j)
    M_sum_T = M_sum.reshape(1, 2)
    B_inv = np.linalg.inv(B)

    a = np.matmul(M_dif_T, B_inv)
    b = -0.5 * np.matmul(np.matmul(M_sum_T, B_inv), M_dif) + threshold  # page 31 случай 2

    # print(a)
    # print(x)

    counter = 0

    for xi in x:
        d_lj = np.matmul(a[0], xi.reshape(2, 1)) + b
        if d_lj < 0:
            counter += 1

    # print(counter)
    return counter / x.shape[0]


def boundary_of_bayes_classifier_for_N(x, M_l, M_j, B_l, B_j, threshold):
    M_l_T = M_l.reshape(1, 2)
    M_j_T = M_j.reshape(1, 2)

    B_l_inv = np.linalg.inv(B_l)
    B_j_inv = np.linalg.inv(B_j)

    det_B_l = np.linalg.det(B_l)
    det_B_j = np.linalg.det(B_j)

    # d_lj = d_l - d_j = 0
    # d_lj = ... = xTAx+bx+c=0
    A = 0.5 * (B_j_inv - B_l_inv)
    b = np.matmul(M_l_T, B_l_inv) - np.matmul(M_j_T, B_j_inv)
    c = (0.5 * np.matmul(np.matmul(M_j_T, B_j_inv), M_j) - 0.5 * np.matmul(np.matmul(M_l_T, B_l_inv), M_l)
         + 0.5 * np.log(det_B_j / det_B_l) + threshold)  # конец 4 лекции

    # print(f"A = {A}")
    # print(f"b = {b}")
    # print(f"c = {c}")

    boundary = []
    for x0 in x:
        coef_a = A[1, 1]
        coef_b = (A[1, 0] + A[0, 1]) * x0 + b[0, 1]
        coef_c = A[0, 0] * x0 ** 2 + b[0, 0] * x0 + c[0]
        D = coef_b ** 2 - 4 * coef_a * coef_c
        if D > 0:
            x1_0 = (-coef_b + np.sqrt(D)) / (2 * coef_a)
            x1_1 = (-coef_b - np.sqrt(D)) / (2 * coef_a)
            boundary.append([x0, x1_0])
            boundary.append([x0, x1_1])
        elif D == 0:
            x1_0 = -coef_b / (2 * coef_a)
            boundary.append([x0, x1_0])

    return np.array(boundary)


def get_erroneous_classification_probabilities(M_l, M_j, B):
    distance = Mahalanobis_distance(M_l, M_j, B)
    print(f"Расстояние Махаланобиса: {distance}")
    p_0 = scipy.stats.norm.cdf(-0.5 * np.sqrt(distance))
    p_1 = 1 - scipy.stats.norm.cdf(0.5 * np.sqrt(distance))
    print(f"Ошибка первого рода: {p_0}")
    print(f"Ошибка второго рода: {p_1}")
    # R = 1/2(p_0+p_1) = 1 - Ф(0.5*sqrt(distance))
    print(f"Суммарный вероятность ошибочной классификации: {p_0 + p_1}")


def experimental_probability_error(x, M_l, M_j, B_l, B_j):
    count = 0

    calc_d = lambda vec, M, B, P: (np.log(P) - np.log(np.sqrt(np.linalg.det(B))) -
                                   0.5 * np.matmul(np.matmul((vec - M), np.linalg.inv(B)), (vec - M).reshape(2, 1))
                                   )  # page 29

    for xi in x:
        d_l = calc_d(xi, M_l, B_l, 0.5)
        d_j = calc_d(xi, M_j, B_j, 0.5)

        if d_j > d_l:
            count += 1

    return count / x.shape[0]


def get_eps(p, N):
    return np.sqrt((1 - p) / (N * p))


def get_N(p, err):
    return (1 - p) / (err ** 2 * p)


def task_1():
    # Data with same B
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    x = np.linspace(min_value, max_value, 100)
    threshold = np.log(0.5 / 0.5)
    y = boundary_of_bayes_classifier_for_N_with_same_B(x, M_0, M_1, B_0, threshold)

    # plt.suptitle("Байесовский классификатор")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y, color="red")
    # plt.show()

    get_erroneous_classification_probabilities(M_0, M_1, B_0)

    exper = experimental_probability_error(sample_1, M_0, M_1, B_0, B_0, 0.5, 0.5)
    print(f"Экспериментальная вероятность для БК {exper}")

    return x, y


def task_2():
    # Data with same B
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    x = np.linspace(min_value, max_value, 100)

    # MinMax classifier
    # p_0 = p_1 => Laplace functions equal => lambda = 0 => P(omega0) = P(omega1)
    P_0 = 0.5
    P_1 = 0.5
    threshold = np.log(P_0 / P_1)
    y_0 = boundary_of_bayes_classifier_for_N_with_same_B(x, M_0, M_1, B_0, threshold)

    # plt.suptitle("Минимаксный классификатор")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y_0, color="red")
    # plt.show()

    # Neyman-Pearson classifier
    p_0 = 0.05
    distance = Mahalanobis_distance(M_0, M_1, B_0)
    # inv Laplace function for (1-p_0): p_0 = 0.05
    inv_laplace = 1.645
    lambda_tilda = -0.5 * distance + np.sqrt(distance) * inv_laplace
    y_1 = boundary_of_bayes_classifier_for_N_with_same_B(x, M_0, M_1, B_0, lambda_tilda)

    # plt.suptitle("Классификатор Неймана-Пирсона")
    # plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    # plt.plot(x, y_1, color="red")
    # plt.show()

    print(f"Экспериментальная верояность для Неймана-Пирсона: {get_count_fail(sample_1, M_0, M_1, B_0, lambda_tilda)}")

    return [(x, y_0), (x, y_1)]


def task_3():
    sample_1 = np.load("Files/arrayX3_1.npy")
    sample_2 = np.load("Files/arrayX3_2.npy")
    sample_3 = np.load("Files/arrayX3_3.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)
    sample_3 = np.transpose(sample_3)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]), np.min(sample_3[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]), np.max(sample_3[:, 0]))

    # x = np.linspace(-2, 2, 200)
    x = np.linspace(min_value, max_value, 200)

    boundary_01 = boundary_of_bayes_classifier_for_N(x, M_0, M_1, B_1, B_2, 0)
    boundary_02 = boundary_of_bayes_classifier_for_N(x, M_0, M_2, B_1, B_3, 0)
    boundary_12 = boundary_of_bayes_classifier_for_N(x, M_1, M_2, B_2, B_3, 0)

    plt.ylim(-2, 3)
    plt.xlim(min_value, max_value)

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='purple', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    plt.plot(sample_3[:, 0], sample_3[:, 1], color='blue', linestyle='none', marker='*')
    plt.scatter(boundary_01[:, 0], boundary_01[:, 1], color="red", s=[5])
    plt.scatter(boundary_02[:, 0], boundary_02[:, 1], color="red", s=[5])
    plt.scatter(boundary_12[:, 0], boundary_12[:, 1], color="red", s=[5])
    plt.show()

    p = experimental_probability_error(sample_1, M_0, M_1, B_1, B_2, 0.5, 0.5)
    print(f"Экспериментальные вероятности ошибочной классификации: {p}")

    eps = get_eps(p, sample_1.shape[0])
    print(f"Относительная погрешность: {eps}")

    n = get_N(p, 0.05)
    print(f"Объем обучающей выборки, обеспечивающий"
          f"получение оценок вероятностей ошибочной классификации"
          f"с погрешностью не более 5%: {n}")


def plot_task3(x_1, y_1_1, y_1_2, y_1_3):
    sample_1 = np.transpose(np.load("Files/arrayX2_1.npy"))
    sample_2 = np.transpose(np.load("Files/arrayX2_2.npy"))
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(15, 7)
    ax[0].set_title("Байесовский классификатор")
    ax[0].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[0].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[0].plot(x_1, y_1_1, color="red")
    ax[1].set_title("Минимаксный классификатор")
    ax[1].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[1].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[1].plot(x_1, y_1_2, color="red")
    ax[2].set_title("Классификатор Неймана-Пирсона")
    ax[2].plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    ax[2].plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')
    ax[2].plot(x_1, y_1_3, color="red")
    plt.show()


def show_all_borders(borders):
    sample_1 = np.load("Files/arrayX2_1.npy")
    sample_2 = np.load("Files/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    plt.plot(borders[0][0], borders[0][1], color="red")
    plt.plot(borders[1][0], borders[1][1], color="purple", linestyle = '-.')
    plt.plot(borders[2][0], borders[2][1], color="pink")

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')

    plt.show()


def main():
    x_0, y_0 = task_1()
    value = task_2()
    x_1, y_1 = value[0]
    x_2, y_2 = value[1]
    # plot_task3(x_0, y_0, y_1, y_2)

    borders = [[x_0, y_0], [x_1, y_1], [x_2, y_2]]
    show_all_borders(borders)

    task_3()


if __name__ == "__main__":
    main()