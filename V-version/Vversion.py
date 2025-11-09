import numpy as np
import matplotlib.pyplot as plt

from Lab_2 import (boundary_of_bayes_classifier_for_N_with_same_B, boundary_of_bayes_classifier_for_N,
                         get_erroneous_classification_probabilities, experimental_probability_error)


M_0 = np.array([0, 0])
M_1 = np.array([-1, 1])

B_0 = np.array([[0.35, 0.15], [0.15, 0.35]])  # Для одинаковых

B_1 = np.array([[0.45, 0.15], [0.15, 0.45]])  # Для различных
B_2 = np.array([[0.15, 0.02], [0.02, 0.15]])


def get_errors_first(sample_1, sample_2, weights, wN):
    counter_0 = 0
    counter_1 = 0

    W = weights[0:2]

    for x in sample_1:
        if W.T @ x + wN > 0:
            counter_0 += 1

    for x in sample_2:
        if W.T @ x + wN < 0:
            counter_1 += 1

    print(f"Ошибка первого рода: {counter_0 / sample_1.shape[0]}")
    print(f"Ошибка второго рода: {counter_1 / sample_2.shape[0]}")


def get_errors(sample_1, sample_2, W_POPOL):
    counter_0 = 0
    counter_1 = 0

    W = W_POPOL[0:2]
    wN = W_POPOL[2]

    for x in sample_1:
        if W.T @ x + wN > 0:
            counter_0 += 1

    for x in sample_2:
        if W.T @ x + wN < 0:
            counter_1 += 1

    print(f"Ошибка первого рода: {counter_0 / sample_1.shape[0]}")
    print(f"Ошибка второго рода: {counter_1 / sample_2.shape[0]}")


def show_all_borders(borders):
    sample_1 = np.load("data/X2_equal.npy")
    sample_2 = np.load("data/X2_equal.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    plt.plot(borders[0][0], borders[0][1], color="red")
    plt.plot(borders[1][0], borders[1][1], color="purple", linestyle='-.')

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')

    plt.show()


def show_all_borders_three(borders):
    sample_1 = np.load("data/arrayX2_1.npy")
    sample_2 = np.load("data/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    plt.plot(borders[0][0], borders[0][1], color="red")
    plt.plot(borders[1][0], borders[1][1], color="purple", linestyle='-.')
    plt.plot(borders[2][0], borders[2][1], color="black", linestyle=':')

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')

    plt.show()


def task_1_same_B():
    sample_1 = np.transpose(np.load("data/arrayX2_1.npy"))
    sample_2 = np.transpose(np.load("data/arrayX2_2.npy"))

    M_dif = M_1 - M_0
    M_sum = M_1 + M_0

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    x = np.linspace(min_value, max_value, 200)

    threshold_1 = -0.5 * (M_dif @ np.linalg.inv(B_0) @ np.transpose(M_sum))  # порог для равных корреляционных матриц
    weights = np.linalg.inv(B_0) @ M_dif

    y_1 = np.array((-threshold_1 - weights[0] * x) / weights[1])

    threshold_2 = np.log(0.5 / 0.5)
    y_2 = boundary_of_bayes_classifier_for_N_with_same_B(x, M_0, M_1, B_0, threshold_2)  # БК

    get_errors_first(sample_1, sample_2, weights, threshold_1)

    return x, y_1, x, y_2


def task_1_different_B():
    sample_3 = np.transpose(np.load("data/arrayX3_1.npy"))
    sample_4 = np.transpose(np.load("data/arrayX3_2.npy"))

    M_dif = M_1 - M_0
    M_dif_T = np.transpose(M_1 - M_0)

    min_value = min(np.min(sample_3[:, 0]), np.min(sample_4[:, 0]))
    max_value = max(np.max(sample_3[:, 0]), np.max(sample_4[:, 0]))

    x = np.linspace(min_value, max_value, 200)

    boundary_01 = boundary_of_bayes_classifier_for_N(x, M_0, M_1, B_1, B_2, 0)  # БК

    weights = np.linalg.inv(0.5 * (B_1 + B_2)) @ M_dif_T  # Фишер
    sigma_1 = np.transpose(weights) @ B_1 @ weights
    sigma_2 = np.transpose(weights) @ B_2 @ weights
    threshold = -1/(sigma_1 + sigma_2) * M_dif @ np.linalg.inv(0.5 * (B_1 + B_2)) @ ((sigma_1 * M_1) + (sigma_2 * M_0))
    boundary_02 = np.array((-threshold - weights[0] * x) / weights[1])

    get_errors_first(sample_3, sample_4, weights, threshold)

    plt.ylim(-2, 3)
    plt.xlim(min_value, max_value)

    plt.plot(sample_3[:, 0], sample_3[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_4[:, 0], sample_4[:, 1], color='green', linestyle='none', marker='*')
    plt.scatter(boundary_01[:, 0], boundary_01[:, 1], color="red", s=[5])
    plt.plot(x, boundary_02, color="purple", linestyle='-.')

    plt.show()


def task_2_same_B():
    sample_1 = np.load("data/arrayX2_1.npy")
    sample_2 = np.load("data/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    class_0_size = np.shape(sample_1)
    class_1_size = np.shape(sample_2)
    z0 = np.concatenate((sample_1, np.ones((class_0_size[0], 1))), axis=1)
    z0 = -1 * z0
    z1 = np.concatenate((sample_2, np.ones((class_1_size[0], 1))), axis=1)

    U = np.transpose(np.concatenate((z0, z1), axis=0))
    Y = np.ones((class_0_size[0] + class_1_size[0], 1))

    inv = np.linalg.inv(U @ np.transpose(U))
    weights = inv @ U @ Y

    x = np.linspace(min_value, max_value, 200)
    y = np.array((-weights[2] - weights[0] * x) / weights[1])

    get_errors(sample_1, sample_2, weights)

    return x, y


def task_2_different_B():
    sample_3 = np.transpose(np.load("data/arrayX3_1.npy"))
    sample_4 = np.transpose(np.load("data/arrayX3_2.npy"))

    M_dif = M_1 - M_0
    M_dif_T = np.transpose(M_1 - M_0)

    min_value = min(np.min(sample_3[:, 0]), np.min(sample_4[:, 0]))
    max_value = max(np.max(sample_3[:, 0]), np.max(sample_4[:, 0]))

    x = np.linspace(min_value, max_value, 200)

    boundary_01 = boundary_of_bayes_classifier_for_N(x, M_0, M_1, B_1, B_2, 0)  # БК

    weights = np.linalg.inv(0.5 * (B_1 + B_2)) @ M_dif_T  # Фишер
    sigma_1 = np.transpose(weights) @ B_1 @ weights
    sigma_2 = np.transpose(weights) @ B_2 @ weights
    threshold = -1/(sigma_1 + sigma_2) * M_dif @ np.linalg.inv(0.5 * (B_1 + B_2)) @ ((sigma_1 * M_1) + (sigma_2 * M_0))
    boundary_02 = np.array((-threshold - weights[0] * x) / weights[1])

    class_0_size = np.shape(sample_3)  # Линейный для уменьшения СКО
    class_1_size = np.shape(sample_4)
    z0 = np.concatenate((sample_3, np.ones((class_0_size[0], 1))), axis=1)
    z0 = -1 * z0
    z1 = np.concatenate((sample_4, np.ones((class_1_size[0], 1))), axis=1)

    U = np.transpose(np.concatenate((z0, z1), axis=0))
    Y = np.ones((class_0_size[0] + class_1_size[0], 1))

    inv = np.linalg.inv(U @ np.transpose(U))
    weights = inv @ U @ Y

    y = np.array((-weights[2] - weights[0] * x) / weights[1])

    get_errors(sample_3, sample_4, weights)

    plt.ylim(-2, 3)
    plt.xlim(min_value, max_value)

    plt.plot(sample_3[:, 0], sample_3[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_4[:, 0], sample_4[:, 1], color='green', linestyle='none', marker='*')
    plt.scatter(boundary_01[:, 0], boundary_01[:, 1], color="red", s=[5])
    plt.plot(x, boundary_02, color="purple", linestyle='-.')
    plt.plot(x, y, color='black', linestyle=':')

    plt.show()


def show_borders(W_arr, x, sample_1, sample_2):
    size = np.array(W_arr).shape

    plt.plot(sample_1[:, 0], sample_1[:, 1], color='blue', linestyle='none', marker='.')
    plt.plot(sample_2[:, 0], sample_2[:, 1], color='green', linestyle='none', marker='*')

    W = W_arr[0]

    y = np.array((-W[2] - W[0] * x) / W[1])

    plt.plot(x, y, color='black', linewidth=3)

    for i in range(1, size[0], 100):
        W = W_arr[i]
        y = np.array((-W[2] - W[0] * x) / W[1])
        plt.plot(x, y, linewidth=1)

    W = W_arr[-1]
    y = np.array((-W[2] - W[0] * x) / W[1])
    plt.plot(x, y, linewidth=3)
    plt.ylim(-4, 3)

    plt.show()

def task_3_same_B():
    sample_1 = np.load("data/arrayX2_1.npy")
    sample_2 = np.load("data/arrayX2_2.npy")
    sample_1 = np.transpose(sample_1)
    sample_2 = np.transpose(sample_2)

    class_0_size = np.shape(sample_1)
    class_1_size = np.shape(sample_2)

    min_value = min(np.min(sample_1[:, 0]), np.min(sample_2[:, 0]))
    max_value = max(np.max(sample_1[:, 0]), np.max(sample_2[:, 0]))

    bet = 0.8

    #weights = np.zeros((3, 1))
    #weights = np.ones((3, 1))
    weights = np.array([[-1], [0], [0]])
    weights_arr = []

    x_s = np.concatenate((sample_1, sample_2), axis=0)
    x_s = np.concatenate((x_s, np.ones((class_0_size[0] + class_1_size[0], 1))), axis=1)
    x_s = np.transpose(x_s)

    r = np.concatenate((np.full(class_0_size[0], -1), np.ones(class_1_size[0])))
    ind = np.arange(class_0_size[0] + class_1_size[0])
    np.random.shuffle(ind)

    counter = 0
    sgn_prev = 0
    alph = 1 / np.power(1, bet)

    for _ in range(5):
        for k in range(class_0_size[0] + class_1_size[0]):
            x_k = x_s[:, ind[k]].reshape((class_0_size[1] + 1, 1))
            d = np.matmul(np.transpose(weights), x_k)[0]

            # if ((d[0] < 0) and (r[ind[k]] > 0)) or ((d[0] >= 0) and (r[ind[k]] < 0)):
            sgn = np.sign(r[ind[k]] - d)[0]

                # if sgn != sgn_prev:
            counter += 1
            sgn_prev = sgn
            alph = 1 / np.power(counter + 1, bet)

            weights = weights + alph * x_k * sgn
            weights_arr.append(weights)

        np.random.shuffle(ind)

    x = np.linspace(min_value, max_value, 200)
    y = np.array((-weights[2] - weights[0] * x) / weights[1])

    threshold_2 = np.log(0.5 / 0.5)
    y_2 = boundary_of_bayes_classifier_for_N_with_same_B(x, M_0, M_1, B_0, threshold_2)  # БК

    get_errors(sample_1, sample_2, weights)

    show_borders(weights_arr, x, sample_1, sample_2)

    return x, y, x, y_2


def task_3_different_B():
    sample_3 = np.transpose(np.load("data/arrayX3_1.npy"))
    sample_4 = np.transpose(np.load("data/arrayX3_2.npy"))

    class_0_size = np.shape(sample_3)
    class_1_size = np.shape(sample_4)

    min_value = min(np.min(sample_3[:, 0]), np.min(sample_4[:, 0]))
    max_value = max(np.max(sample_3[:, 0]), np.max(sample_4[:, 0]))

    x = np.linspace(min_value, max_value, 200)

    boundary_01 = boundary_of_bayes_classifier_for_N(x, M_0, M_1, B_1, B_2, 0)  # БК

    bet = 0.8

    #weights = np.zeros((3, 1))
    #weights = np.ones((3, 1))
    weights = np.array([[-1], [0], [0]])
    weights_arr = []

    x_s = np.concatenate((sample_3, sample_4), axis=0)
    x_s = np.concatenate((x_s, np.ones((class_0_size[0] + class_1_size[0], 1))), axis=1)
    x_s = np.transpose(x_s)

    r = np.concatenate((np.full(class_0_size[0], -1), np.ones(class_1_size[0])))
    ind = np.arange(class_0_size[0] + class_1_size[0])
    np.random.shuffle(ind)

    counter = 0
    sgn_prev = 0
    alph = 1 / np.power(1, bet)

    for _ in range(5):
        for k in range(class_0_size[0] + class_1_size[0]):
            x_k = x_s[:, ind[k]].reshape((class_0_size[1] + 1, 1))
            d = np.matmul(np.transpose(weights), x_k)[0]

            #if ((d[0] < 0) and (r[ind[k]] > 0)) or ((d[0] >= 0) and (r[ind[k]] < 0)):
            sgn = np.sign(r[ind[k]] - d)[0]

                #if sgn != sgn_prev:
            counter += 1
            sgn_prev = sgn
            alph = 1 / np.power(counter + 1, bet)

            weights = weights + alph * x_k * sgn
            weights_arr.append(weights)

        np.random.shuffle(ind)

    y = np.array((-weights[2] - weights[0] * x) / weights[1])

    get_errors(sample_3, sample_4, weights)

    show_borders(weights_arr, x, sample_3, sample_4)

    # plt.ylim(-2, 3)
    # plt.xlim(min_value, max_value)
    #
    # plt.plot(sample_3[:, 0], sample_3[:, 1], color='blue', linestyle='none', marker='.')
    # plt.plot(sample_4[:, 0], sample_4[:, 1], color='green', linestyle='none', marker='*')
    # plt.scatter(boundary_01[:, 0], boundary_01[:, 1], color="red", s=[5])
    # plt.plot(x, y, color='purple', linestyle=':')
    #
    # plt.show()



def task_1():
    x_0, y_0, x_1, y_1 = task_1_same_B()

    borders = [[x_0, y_0], [x_1, y_1]]
    show_all_borders(borders)  # для одинаковых

    task_1_different_B()


def task_2():
    x_0, y_0 = task_2_same_B()
    x_1, y_1, x_2, y_2 = task_1_same_B()

    borders = [[x_1, y_1], [x_2, y_2], [x_0, y_0]]
    show_all_borders_three(borders)

    task_2_different_B()


def task_3():
    x_0, y_0, x_1, y_1 = task_3_same_B()

    borders = [[x_0, y_0], [x_1, y_1]]
    # show_all_borders(borders)

    task_3_different_B()


def main():
    #task_1()
    #task_2()
    task_3()


if __name__ == "__main__":
    main()