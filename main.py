import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm
from qpsolvers import solve_qp

path_to_first_file = 'Files/arrayX2_1.npy'
path_to_second_file = 'Files/arrayX2_2.npy'

path_to_first_file_lab_1 = "Lab_1/arrayX3_1.npy"
path_to_second_file_lab_1 = "Lab_1/arrayX3_2.npy"

M0 = np.array([[-4], [-1]])  # mean vector
B0 = np.array([[1.5, -0.4], [-0.4, 1]])  # covariance matrix

M1 = np.array([[2], [4]])  # mean vector
B1 = np.array([[1.5, 1], [1, 2]])  # covariance matrix


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


M0_BAYES = np.array([0, 0])
M1_BAYES = np.array([1, 1])

# Task 2 from lab 1
B0_BAYES = np.array([[0.1, 0.05], [0.05, 0.1]])
B1_BAYES = np.array([[0.1, 0.01], [0.01, 0.1]])


def load_data_from_1_lab():
    z0 = np.load(path_to_first_file_lab_1).T
    z1 = np.load(path_to_second_file_lab_1).T
    x = np.concatenate((z0, z1), axis=1)
    return z0, z1, x


def generate_random_vectors(mean_vector, covariance_matrix, N, saved_file_name):
    # transformation matrix for generation
    A = np.zeros((2,2))
    A[0, 0] = math.sqrt(covariance_matrix[0, 0])
    A[1, 0] = covariance_matrix[1, 0]/math.sqrt(covariance_matrix[0, 0])
    A[1, 1] = math.sqrt(covariance_matrix[1, 1] - covariance_matrix[0, 1] * covariance_matrix [0, 1] / covariance_matrix[0, 0])

    vectorsNorm01 = np.random.randn(2, N)  # (0,1)-normal vectors

    # generated vectors with given mean M and covariance B
    x = np.matmul(A, vectorsNorm01) + np.repeat(mean_vector, N, 1)
    np.save(saved_file_name, x)

    # plt.plot(x[0,:], x[1,], color='green', marker='X', linestyle='none')
    # plt.show()


def get_q(data):
    return np.full(data.shape[1], -1)


def get_P(data, r):
    matrix_size = data.shape[1]

    P = np.empty((matrix_size, matrix_size))
    for j in range(matrix_size):
        for i in range(matrix_size):
            r_ji = r[j] * r[i]
            # print(f"r {r_ji}")
            x_ji = np.matmul(data[:, j], data[:, i])
            P[j, i] = r_ji * x_ji
    # print(P)
    # print(P.shape)
    return P


def get_K(x, y, K, K_params):
    if K == 'poly':
        gamma = K_params[0]
        r = K_params[1]
        d = K_params[2]
        # if x.shape != (1, 2):
        #     x = x.reshape(1, 2)
        tmp = np.matmul(x, y) * gamma + r
        return np.power(tmp, d)
    if K == 'rbf':
        gamma = K_params[0]
        # print(f"x {x}")
        # print(f"y {y}")
        # print(f"gamma {gamma}")
        # print(f"руками {np.exp(-gamma * np.sum(np.power((x - y), 2)))}")
        # print(f"norm {np.exp(-gamma * np.linalg.norm(x - y) ** 2)}")
        return np.exp(-gamma * np.sum(np.power((x - y), 2)))
    if K == 'sigmoid':
        # if x.shape != (1, 2):
        #     x = x.reshape(1, 2)
        gamma = K_params[0]
        r = K_params[1]
        # print(f"x {x}")
        # print(f"y {y}")
        # print(np.tanh(gamma * np.matmul(x, y) + r))
        return np.tanh(gamma * np.matmul(x, y) + r)


def get_P_kernel(data, r, K, K_params):
    matrix_size = data.shape[1]

    P = np.empty((matrix_size, matrix_size))
    for j in range(matrix_size):
        for i in range(matrix_size):
            r_ji = r[j] * r[i]
            K_value = get_K(data[:, j], data[:, i], K=K, K_params=K_params)
            P[j, i] = r_ji * K_value

    return P


def get_qp_solves(x, r, C=None, K=None):
    q = np.full((x.shape[1], ), -1, 'd')
    if K is None:
        P = get_P(x, r)
    else:
        k = K[0]
        k_params = K[1]
        P = get_P_kernel(x, r, K=k, K_params=k_params)

    A = r
    b = np.array([0], dtype='d')
    if C is None:
        G = np.eye(x.shape[1]) * -1
        h = np.zeros(x.shape[1])
    else:
        G = np.concatenate((np.eye(x.shape[1]) * -1, np.eye(x.shape[1])), axis=0)
        h = np.concatenate((np.zeros((x.shape[1],)), np.full((x.shape[1],), C)), axis=0)

    # print(f"P: {P}")
    # print(f"A: {A}")
    # print(f"q: {q}")
    # print(f"b: {b}")
    # print(f"G: {G}")
    # print(f"h: {h}")

    lambdas = solve_qp(P, q, G, h, A, b, solver="cvxopt")
    return lambdas


def get_support_vectors(x, lambdas, r, eps=1e-10):
    indexes = np.where(lambdas >= eps)
    sv = x[:, indexes[0]]
    sv_lambdas = lambdas[indexes[0]]
    sv_r = r[indexes[0]]
    return sv, sv_lambdas, sv_r


# poly: gamma, r, d  |  (gamma * x1 * x2 + r)^d
# rbf: gamma  |   exp(-gamma * ||x1 - x2||)
# sigmoid: gamma, r     |   tanh(gamma * x1 * x2 + r)
# gausse: тот же rbf, в котором gamma = 1/(2*sigma^2)
def get_lib_svm(C=None, K=None, K_params=None):
    if K is None and C is None:
        return svm.LinearSVC(dual="auto")
    if K == "linear":
        if C is None:
            return svm.SVC(kernel=K)
        else:
            return svm.SVC(C=C, kernel=K)  # linear version of SVM. Instead of svm.SVC()
    if K == 'poly':
        return svm.SVC(C=C, kernel=K, degree=K_params[2], coef0=K_params[1], gamma=K_params[0])
    if K == 'rbf':
        return svm.SVC(C=C, kernel=K, gamma=K_params[0])
    if K == 'sigmoid':
        return svm.SVC(C=C, kernel=K, coef0=K_params[1], gamma=K_params[0])


def get_coef_for_linear(x, clf):
    x = x.transpose()
    N = x.shape[0] // 2

    y_ideal = np.zeros((2 * N))
    y_ideal[N:2 * N] = 1

    clf.fit(x, y_ideal)
    return np.transpose(clf.coef_), clf.intercept_[0]


def get_border_for_linear(W, wN, x, d_value=0):
    assert W.shape == (2, 1)

    eps = 0.001
    if np.abs(W[1, 0]) <= eps:
        x1 = np.copy(x)
        x0 = np.array((d_value - wN - W[1, 0] * x1) / W[0, 0])
    else:
        x0 = np.copy(x)
        x1 = np.array((d_value - wN - W[0, 0] * x0) / W[1, 0])
    return x0, x1


def show_plots_for_linear(z0, z1, hyperplane_1, hyperplane_2):

    plt.scatter(z0[0, :], z0[1, :], color='red')  # plot generated data
    plt.scatter(z1[0, :], z1[1, :], color='blue')  # plot generated data

    for border in hyperplane_1:
        plt.plot(border[0], border[1], color='black', marker='D', markersize=8)

    for border in hyperplane_2:
        plt.plot(border[0], border[1], color='green', marker='o')

    plt.show()


def get_d_value(support_vector, x, lambdas, r, K, K_params):
    values = lambdas * r * [get_K(support_vector[:, i], x, K, K_params) for i in range(support_vector.shape[1])]
    return np.sum(values)


def calculate_p_errors(W, wN, x, r):
    assert x.shape[1] == r.shape[0]

    counter_p0 = 0
    counter_p1 = 0
    N = x.shape[1] / 2  # не всегда

    if W.shape == (2, 1):
        W = W.reshape(1, 2)[0]

    d_values = np.matmul(W, x) + wN
    for i in range(d_values.shape[0]):
        if d_values[i] > 0 and r[i] == -1:
            counter_p0 += 1
        if d_values[i] < 0 and r[i] == 1:
            counter_p1 += 1

    return counter_p0/N, counter_p1/N


def calculate_p_errors_with_K(x, r, support_vector, lambdas, r_sv, K, K_params, wN):
    assert x.shape[1] == r.shape[0]

    counter_p0 = 0
    counter_p1 = 0
    N = x.shape[1] / 2  # не всегда

    for i in range(x.shape[1]):
        if get_d_value(support_vector, x[:, i], lambdas, r_sv, K, K_params) + wN > 0 and r[i] == -1:
            counter_p0 += 1
        if get_d_value(support_vector, x[:, i], lambdas, r_sv, K, K_params) + wN < 0 and r[i] == 1:
            counter_p1 += 1

    return counter_p0/N, counter_p1/N


def task_1():
    N = 100

    generate_random_vectors(M0, B0, N, path_to_first_file)
    generate_random_vectors(M1, B1, N, path_to_second_file)

    # check save data
    z0 = np.load(path_to_first_file)
    z1 = np.load(path_to_second_file)
    plt.plot(z0[0, :], z0[1, :], color='red', marker='.', linestyle='none')  # plot generated data
    plt.plot(z1[0, :], z1[1, :], color='blue', marker='.', linestyle='none')  # plot generated data
    plt.show()


def task_2():
    z0 = np.load(path_to_first_file)
    z1 = np.load(path_to_second_file)
    x = np.concatenate((z0, z1), axis=1)

    r = np.ones(x.shape[1])
    r[0:z0.shape[1]] = -1

    lambdas = get_qp_solves(x, r)
    sv, sv_lambda, sv_r = get_support_vectors(x, lambdas, r)

    W = np.sum(sv_lambda * sv_r * sv, axis=1)
    wN = np.sum(sv_r - np.matmul(W, sv)) / sv.shape[1]

    x_min_max_value = [np.min(x[0, :]), np.max(x[0, :])]

    border_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value)
    border_0_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value, d_value=-1)
    border_1_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value, d_value=1)
    hyperplane_qp = [border_qp, border_0_qp, border_1_qp]

    # =================================================================

    clf = get_lib_svm(K="linear")
    W_svm, wN_svm = get_coef_for_linear(x, clf)

    border_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value)
    border_0_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value, d_value=-1)
    border_1_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value, d_value=1)
    hyperplane_svm = [border_svm, border_0_svm, border_1_svm]

    # =================================================================

    clf = get_lib_svm()
    W_svm_LinearSVC, wN_svm_LinearSVC = get_coef_for_linear(x, clf)

    border_svm_LinearSVC = get_border_for_linear(W_svm_LinearSVC, wN_svm_LinearSVC, x_min_max_value)
    border_0_svm_LinearSVC = get_border_for_linear(W_svm_LinearSVC, wN_svm_LinearSVC, x_min_max_value, d_value=-1)
    border_1_svm_LinearSVC = get_border_for_linear(W_svm_LinearSVC, wN_svm_LinearSVC, x_min_max_value, d_value=1)
    hyperplane_svm_LinearSVC = [border_svm_LinearSVC, border_0_svm_LinearSVC, border_1_svm_LinearSVC]

    # =================================================================

    plt.suptitle("QP solve vs svm.SVC")
    plt.scatter(sv[0, :], sv[1, :], marker="x", s=150)
    show_plots_for_linear(z0, z1, hyperplane_qp, hyperplane_svm)

    plt.suptitle("QP solve vs svm.LinearSVC")
    plt.scatter(sv[0, :], sv[1, :],  marker="x", s=150)
    show_plots_for_linear(z0, z1, hyperplane_qp, hyperplane_svm_LinearSVC)

    plt.suptitle("svm.SVC vs svm.LinearSVC")
    plt.scatter(sv[0, :], sv[1, :], marker="x", s=150)
    show_plots_for_linear(z0, z1, hyperplane_svm, hyperplane_svm_LinearSVC)

    # =================================================================

    p0, p1 = calculate_p_errors(W, wN, x, r)
    print("Ошибки для метода решения квадратичных задач")
    print(f"\tОшибка первого рода: {p0}")
    print(f"\tОшибка второго рода: {p1}")

    p0_svm, p1_svm = calculate_p_errors(W_svm, wN_svm, x, r)
    print("\n\nОшибки для метода sklearn.svm.SVC")
    print(f"\tОшибка первого рода: {p0_svm}")
    print(f"\tОшибка второго рода: {p1_svm}")

    p0_svm_LinearSVC, p1_svm_LinearSVC = calculate_p_errors(W_svm_LinearSVC, wN_svm_LinearSVC, x, r)
    print("\n\nОшибки для метода LinearSVC")
    print(f"\tОшибка первого рода: {p0_svm_LinearSVC}")
    print(f"\tОшибка второго рода: {p1_svm_LinearSVC}")

    # N = z0.shape[1]
    # y_ideal = np.zeros((2 * N))
    # y_ideal[N:2 * N] = 1
    # y_predicted = clf.predict(x.T)
    # y_dif = np.abs(y_ideal - y_predicted)
    # N_err = np.sum(y_dif)
    # y_dif_01 = y_dif[0:N]
    # y_dif_10 = y_dif[N:2 * N]
    # N01 = np.sum(y_dif_01)
    # N10 = np.sum(y_dif_10)
    #
    # print(N_err / N)
    # print(N01 / N)
    # print(N10 / N)


def task_3():

    z0 = np.load("Files/arrayX2_1.npy")
    z1 = np.load("Files/arrayX2_2.npy")
    x = np.concatenate((z0, z1), axis=1)

    r = np.ones(x.shape[1])
    r[0:z0.shape[1]] = -1

    for C in [0.1, 1, 10]:
        lambdas = get_qp_solves(x, r, C=C)
        sv, sv_lambda, sv_r = get_support_vectors(x, lambdas, r, eps=0.09)
        # print(f"sv: {sv}")
        # print(f"sv_lambda :{sv_lambda}")
        # print(f"sv_r: {sv_r}")

        W = np.sum(sv_lambda * sv_r * sv, axis=1)
        wN = np.sum(sv_r - np.matmul(W, sv)) / sv.shape[1]
        # print(f"W={W}")
        # print(f"wN={wN}")

        x_min_max_value = [np.min(x[0, :]), np.max(x[0, :])]
        # print(x_min_max_value)

        border_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value)
        border_0_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value, d_value=-1)
        border_1_qp = get_border_for_linear(W.reshape(2, 1), wN, x_min_max_value, d_value=1)
        hyperplane_qp = [border_qp, border_0_qp, border_1_qp]

        clf = get_lib_svm(K="linear", C=C)
        W_svm, wN_svm = get_coef_for_linear(x, clf)

        border_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value)
        border_0_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value, d_value=-1)
        border_1_svm = get_border_for_linear(W_svm, wN_svm, x_min_max_value, d_value=1)
        hyperplane_svm = [border_svm, border_0_svm, border_1_svm]

        show_plots_for_linear(z0, z1, hyperplane_qp, hyperplane_svm)

        p0, p1 = calculate_p_errors(W, wN, x, r)
        print("Ошибки для метода решения квадратичных задач")
        print(f"\tОшибка первого рода: {p0}")
        print(f"\tОшибка второго рода: {p1}")

        p0_svm, p1_svm = calculate_p_errors(W_svm, wN_svm, x, r)
        print("\n\nОшибки для метода sklearn.svm.SVC")
        print(f"\tОшибка первого рода: {p0_svm}")
        print(f"\tОшибка второго рода: {p1_svm}\n\n")

        # N = z0.shape[1]
        # y_ideal = np.zeros((2 * N))
        # y_ideal[N:2 * N] = 1
        # y_predicted = clf.predict(x.T)
        # y_dif = np.abs(y_ideal - y_predicted)
        # N_err = np.sum(y_dif)
        # y_dif_01 = y_dif[0:N]
        # y_dif_10 = y_dif[N:2 * N]
        # N01 = np.sum(y_dif_01)
        # N10 = np.sum(y_dif_10)
        #
        # print(N_err / N)
        # print(N01 / N)
        # print(N10 / N)


def task_4():
    z0 = np.load("Files/arrayX2_1.npy")
    z1 = np.load("Files/arrayX2_2.npy")

    x = np.concatenate((z0, z1), axis=1)

    # poly: gamma, r, d  |  (gamma * x1 * x2 + r)^d
    # rbf: gamma  |   exp(-gamma * ||x1 - x2||)
    # sigmoid: gamma, r     |   tanh(gamma * x1 * x2 + r)
    # gausse: тот же rbf, в котором gamma = 1/(2*sigma^2)

    K_poly = ["poly", [1, 3, 2]]
    K_rbf = ["rbf", [1]]
    K_gausse = ["rbf", [1/(2 * 0.7**2)]]  # sigma = 1

    K = K_gausse

    r = np.ones(x.shape[1])
    r[0:z0.shape[1]] = -1

    for C in [0.1, 1, 10]:
        lambdas = get_qp_solves(x, r, C=C, K=K)
        sv, sv_lambda, sv_r = get_support_vectors(x, lambdas, r, eps=0.01)
        # print(f"sv: {sv}")
        # print(f"sv_lambda :{sv_lambda}")
        # print(f"sv_r: {sv_r}")

        tmp = sv_r - np.array([get_d_value(sv, sv[:, j], sv_lambda, sv_r, K[0], K[1]) for j in range(sv.shape[1])])
        wN = np.sum(tmp) / sv.shape[1]
        # print(f"wN: {wN}")

        x0_min_max_value = [np.min(x[0, :]), np.max(x[0, :])]
        x1_min_max_value = [np.min(x[1, :]), np.max(x[1, :])]

        x0 = np.linspace(x0_min_max_value[0], x0_min_max_value[1], 50)
        x1 = np.linspace(x1_min_max_value[0], x1_min_max_value[1], 50)
        x0v, x1v = np.meshgrid(x0, x1)
        x0x1 = np.vstack((x0v.ravel(), x1v.ravel())).T

        d_values = []
        for i in range(x0x1.shape[0]):
            d_values.append(get_d_value(sv, x0x1[i], sv_lambda, sv_r, K[0], K[1]) + wN)
        d_values = np.array(d_values).reshape(x0v.shape)

        plt.scatter(z0[0, :], z0[1, :], color='red')
        plt.scatter(z1[0, :], z1[1, :], color='blue')

        plt.contour(x0v, x1v, d_values, levels=[-1, 0, 1], colors=['black', 'black', 'black'])
        # plt.show()

        clf = get_lib_svm(C, K[0], K[1])

        y_ideal = np.zeros((x.shape[1]))
        y_ideal[z0.shape[1]: x.shape[1]] = 1

        clf.fit(x.T, y_ideal)
        # sv_indexes = clf.support_
        # print(f"sv_indexes {sv_indexes}")

        plt.scatter(z0[0, :], z0[1, :], color='red')
        plt.scatter(z1[0, :], z1[1, :], color='blue')

        d_values_lib = clf.decision_function(x0x1).reshape(x0v.shape)
        plt.contour(x0v, x1v, d_values_lib, levels=[-1, 0, 1], colors=['green', 'green', 'green'])

        boundary_bayes = boundary_of_bayes_classifier_for_N(x0, M0_BAYES, M1_BAYES, B0_BAYES, B1_BAYES, 0)
        plt.scatter(boundary_bayes[:, 0], boundary_bayes[:, 1], color="purple", s=[7])
        plt.ylim(x1_min_max_value[0], x1_min_max_value[1])
        plt.show()

        p0, p1 = calculate_p_errors_with_K(x, r, sv, sv_lambda, sv_r, K[0], K[1], wN)
        print(f"Для метода решения квадратичных задач")
        print(f"\tОшибка первого рода: {p0}")
        print(f"\tОшибка второго рода: {p1}")

        N = z0.shape[1]
        y_predicted = clf.predict(x.T)
        y_dif = np.abs(y_ideal - y_predicted)

        y_dif_01 = y_dif[0:N]
        y_dif_10 = y_dif[N:2 * N]
        p0_svc = np.sum(y_dif_01) / N  # При неправильной классификации записывается 2
        p1_svc = np.sum(y_dif_10) / N

        print(f"\n\nОшибки для метода sklearn.svm.SVC")
        print(f"\tОшибка первого рода: {p0_svc}")
        print(f"\tОшибка второго рода: {p1_svc}")


        # p0_bayes, p1_bayes = get_experimental_p_error(z0.T, z1.T, M0_BAYES, M1_BAYES, B0_BAYES, B1_BAYES)
        # print(f"\n\nДля Байесовского классификатора")
        # print(f"\tОшибка первого рода: {p0_bayes}")
        # print(f"\tОшибка второго рода: {p1_bayes}")
        # print("\n============================================\n")


def main():
    task_1()
    #task_2()
    task_3()
    task_4()


if __name__ == '__main__':
    main()