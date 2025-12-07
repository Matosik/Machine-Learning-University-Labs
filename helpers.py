import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple

def compute_svm_errors_manual(X1: np.ndarray, X2: np.ndarray, w: np.ndarray, b: float) -> Tuple[float, float]:
    """
    Считаем долю ошибочной классификации для двух классов по формулам w^T x + b
    Аналогично compute_number_errors2
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Класс +1 (X1)
    count_errors_I = 0
    for x in X1:
        if np.abs(np.dot(w, x) + b) > np.abs(x[1]):
            count_errors_I += 1
    err_I = count_errors_I / n1

    # Класс -1 (X2)
    count_errors_II = 0
    for x in X2:
        if np.abs(np.dot(w, x) + b) < np.abs(x[1]):
            count_errors_II += 1
    err_II = count_errors_II / n2

    return err_I, err_II

def compute_svm_errors_kernel(X1: np.ndarray, X2: np.ndarray, kernel_info: dict) -> Tuple[float, float]:
    """
    Вычисляет долю ошибочных классификаций для двух классов по ядру SVM.
    
    X1: объекты класса +1
    X2: объекты класса -1
    kernel_info: словарь из solve_svm_kernel, содержит:
        'X_train', 'r_train', 'lambda_vec', 'support_indices', 'kernel_type', 'kernel_params'
    """
    X_train = kernel_info['X_train']
    r_train = kernel_info['r_train']
    lambda_vec = kernel_info['lambda_vec']
    support_indices = kernel_info['support_indices']
    K_matrix = kernel_info['K_matrix']
    kernel_type = kernel_info['kernel_type']
    kernel_params = kernel_info['kernel_params']
    w_N = kernel_info.get('w_N', 0.0)  # свободный член
    
    # Функция решения SVM для одного объекта
    def decision_value(x):
        # K(x_i, x)
        if kernel_type == 'linear':
            # Линейное ядро: можно использовать w^T x + b
            w = np.sum((lambda_vec[support_indices] * r_train[support_indices])[:, np.newaxis] * X_train[support_indices], axis=0)
            return np.dot(w, x) + w_N
        else:
            K_x = kernel_matrix_single(x, X_train[support_indices], kernel_type, **kernel_params)
            return np.sum(lambda_vec[support_indices] * r_train[support_indices] * K_x) + w_N
    
    # Ошибки класса +1
    count_errors_I = sum(1 for x in X1 if np.sign(decision_value(x)) != 1)
    err_I = count_errors_I / len(X1)
    
    # Ошибки класса -1
    count_errors_II = sum(1 for x in X2 if np.sign(decision_value(x)) != -1)
    err_II = count_errors_II / len(X2)
    
    return err_I, err_II


def kernel_matrix_single(x: np.ndarray, X_support: np.ndarray, kernel_type: str, **kernel_params) -> np.ndarray:
    """
    Вычисляет вектор K(x_j, x) для опорных векторов
    """
    if kernel_type == 'linear':
        return np.dot(X_support, x)
    elif kernel_type == 'poly':
        degree = kernel_params.get('degree', 3)
        return (np.dot(X_support, x) + 1) ** degree
    elif kernel_type == 'rbf':
        gamma = kernel_params.get('gamma', 1.0)
        diff = X_support - x
        return np.exp(-gamma * np.sum(diff**2, axis=1))
    elif kernel_type == 'sigmoid':
        gamma = kernel_params.get('gamma', 0.1)
        coef0 = kernel_params.get('coef0', 0)
        return np.tanh(gamma * np.dot(X_support, x) + coef0)
    else:
        raise ValueError(f"Unknown kernel type {kernel_type}")


def log_gaussian_pdf(X, mu, Sigma):
    d = X.shape[1]
    Sigma_inv = np.linalg.inv(Sigma)
    diff = X - mu
    quad = np.sum((diff @ Sigma_inv) * diff, axis=1)
    logdet = np.linalg.slogdet(Sigma)[1]
    return -0.5 * (quad + logdet + d * np.log(2*np.pi))

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, clf, 
                          title: str, ax=None, plot_margins: bool = True, bayes_params = None):
    """Визуализация разделяющей гиперплоскости и полосы."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Границы графика
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Создаем сетку для построения контуров
    grid_size = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(clf, 'decision_function'):        
        # Вычисляем значения decision_function
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Рисуем контуры
        if plot_margins:
            ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
                    colors='k', linestyles=['--', '-', '--'], 
                    alpha=0.8)
        else:
            ax.contour(xx, yy, Z, levels=[0], 
                    colors='k', linestyles=['-'], 
                    alpha=1.0)
    
    if bayes_params is not None:
        M1 = bayes_params["M1"]
        M2 = bayes_params["M2"]
        B1 = bayes_params["B1"]
        B2 = bayes_params["B2"]

        logp1 = log_gaussian_pdf(grid, M1, B1)
        logp2 = log_gaussian_pdf(grid, M2, B2)

        # равные priors => сравниваем log плотности
        bayes_Z = (logp1 - logp2).reshape(xx.shape)

        # Рисуем контур Байеса
        ax.contour(xx, yy, bayes_Z, levels=[0], colors='green',
                linewidths=2, linestyles='-', alpha=0.8)

        # Создаем ручку для легенды
        bayes_line = Line2D([0], [0], color='green', lw=2, label='Байесовская граница')
        ax.add_line(bayes_line)

    # Цвета для классов
    colors = ['blue', 'orange']
    
    # Отображение точек
    for idx, class_label in enumerate([1, -1]):
        mask = y == class_label
        ax.scatter(X[mask, 0], X[mask, 1], s=30, alpha=0.7, 
                  label=f'Класс {class_label}', color=colors[idx])
    
    # Отображение опорных векторов
    if hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                  s=100, linewidth=1, facecolors='none', edgecolors='red',
                  label='Опорные векторы')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Признак x1')
    ax.set_ylabel('Признак x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return ax

# Создаем классификатор для QP решения
class QPClassifier:
    """Классификатор на основе решения QP."""
    def __init__(self, w, b, X_train=None, lambda_vec=None):
        self.w = w
        self.b = b
        self.X_train = X_train
        self.lambda_vec = lambda_vec

        if X_train is not None and lambda_vec is not None:
            epsilon = 1e-6
            self.support_indices = lambda_vec > epsilon
            self.support_vectors_ = X_train[self.support_indices]
        else:
            self.support_indices = None
            self.support_vectors_ = None
        
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """Для совместимости с sklearn-подобным интерфейсом."""
        return np.sign(self.decision_function(X))

def kernel_matrix(X1, kernel_type='linear', X2=None, **params):
    if X2 is None:
        X2 = X1

    if kernel_type == 'linear':
        return X1 @ X2.T

    elif kernel_type == 'poly':
        degree = params.get('degree', 3)
        coef0 = params.get('coef0', 1)
        return (X1 @ X2.T + coef0) ** degree

    elif kernel_type == 'rbf':
        gamma = params.get('gamma', 1.0)
        X1_sq = np.sum(X1**2, axis=1)[:, None]
        X2_sq = np.sum(X2**2, axis=1)[None, :]
        return np.exp(-gamma * (X1_sq + X2_sq - 2 * X1 @ X2.T))

    elif kernel_type == 'sigmoid':
        gamma = params.get('gamma', 1.0)
        coef0 = params.get('coef0', 0.0)
        return np.tanh(gamma * (X1 @ X2.T) + coef0)

    else:
        raise ValueError(f"Неизвестный тип ядра: {kernel_type}")

class KernelQPClassifier:
    def __init__(self, w_N, kernel_info):
        self.b = w_N
        self.kernel_type = kernel_info['kernel_type']
        self.kernel_params = kernel_info['kernel_params']
        self.X_train = kernel_info['X_train']
        self.r_train = kernel_info['r_train']
        self.lambda_vec = kernel_info['lambda_vec']

        self.support_indices = kernel_info['support_indices']
        self.support_vectors_ = self.X_train[self.support_indices]

    def kernel(self, X1, X2):
        return kernel_matrix(X1, self.kernel_type, **self.kernel_params, X2=X2)

    def decision_function(self, X):
        # K(x_i, X) — матрица размера (N_train × N_test)
        K_eval = kernel_matrix(self.X_train, self.kernel_type, 
                               X2=X,
                               **self.kernel_params)

        return np.dot(self.lambda_vec * self.r_train, K_eval) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))