import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import uuid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 1.1 Расширение линейной регрессии
class LinearRegressionWithRegularization(nn.Module):
    """
    Линейная регрессия с возможностью L1/L2 регуляризации и ранней остановки.
    """
    def __init__(
            self,
            in_features: int,
            out_features: int = 1,
            regularization: str = 'l1',
            l1_lambda: float = 0.01,
            l2_lambda: float = 0.01
    ) -> None:
        """
        Функция инициализации класса линейной регрессии с возможность регуляризации и ранней остановки
        :param in_features: Размер входного признакового пространства.
        :param out_features:Размер выходного пространства. По умолчанию 1.
        :param regularization: Тип регуляризации ('l1' или 'l2'). По умолчанию 'l1'.
        :param l1_lambda: Коэффициент для L1-регуляризации. По умолчанию 0.01.
        :param l2_lambda: Коэффициент для L2-регуляризации. По умолчанию 0.01.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        if regularization not in ('l1', 'l2'):
            raise AttributeError('Регуляризация должна быть равна либо l1, либо l2')
        self.regularization = regularization
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state = None

    def forward(self, x: torch.Tensor) -> nn.Linear:
        """
        Прямой проход модели
        :param x: Входные данные для прямого прохода
        :return: Результаты предсказания модели
        """
        return self.linear(x)

    def regularization_loss(self) -> torch.Tensor:
        """
        Вычисляет регуляризационную потерю
        :return: Суммарная регуляризацонная потеря (L1 или L2)
        """
        reg_loss = torch.tensor(0.)
        for name, param in self.named_parameters():
            if 'weight' in name:
                if self.regularization == 'l1':
                    reg_loss += self.l1_lambda * torch.abs(param).sum()
                elif self.regularization == 'l2':
                    reg_loss += self.l2_lambda * torch.pow(param, 2).sum()
        return reg_loss

    def update_early_stopping(self, val_loss: float, min_delta: float = 1e-4) -> None:
        """
        Обновляет состояние ранней остановки на основе текущей ошибки.
        :param val_loss: Ошибка на валидационной выборке
        :param min_delta: Минимальное улучшение для сброса счетчика. По умолчанию 1e-4.
        :return: None
        """
        if val_loss < self.best_val_loss - min_delta:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            self.best_model_state = self.state_dict().copy()
        else:
            self.epochs_no_improve += 1

    def check_early_stopping(self, patience: int = 5):
        """
        Проверяет, нужно ли остановить обучение.
        :param patience: Число эпох без улучшения для остановки. По умолчанию 5.
        :return: True, если обучение нужно остановить, иначе False.
        """
        if self.epochs_no_improve >= patience:
            if self.best_model_state is not None:
                self.load_state_dict(self.best_model_state)
            return True
        return False

    def evaluate(self, loader: DataLoader, criterion) -> float:
        """
        Оценивает модель на данных из loader и возвращает значение метрики.
        :param loader: Загрузчик данных
        :param criterion: Функция потерь
        :return: Средняя ошибка на наборе данных
        """
        self.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self(X_batch)
                batch_size = X_batch.size(0)
                total_samples += batch_size
                total_loss += criterion(outputs, y_batch).item() * batch_size

        if total_samples == 0:
            logging.warning("Пустой DataLoader")
            return float('nan')

        return total_loss / total_samples

    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            criterion,
            patience: int = 5
    ) -> None:
        """
        Обучает модель с возможномтью ранней остановки.
        :param train_loader: Загрузчик обучающих данных
        :param val_loader: Загрузчик валидационных данных
        :param epochs: Максимальное число эпох
        :param optimizer: Оптимизатор
        :param criterion: Функция потерь
        :param patience: Терпение для ранней остановки. По умолчанию 5.
        :return: None
        """
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch) + self.regularization_loss()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss = self.evaluate(val_loader, criterion)

            logging.info(
                f'Эпоха {epoch + 1}/{epochs} '
                f'| Ошибка на Train: {train_loss:.4f} '
                f'| Ошибка на Val: {val_loss:.4f}'
            )

            self.update_early_stopping(val_loss)
            if self.check_early_stopping(patience):
                logging.info(f'Ранняя остановка на эпохе {epoch + 1}')
                break
        val_loss = self.evaluate(val_loader, criterion)

        logging.info(f'Финальная ошибка на Val: {val_loss:.4f}')


# 1.2 Расширение логистической регрессии
class LogisticRegressionMultiClass(nn.Module):
    """
    Логистическая регрессия с возможность мультиклассовой классификации
    """
    def __init__(self, in_features: int, num_classes: int) -> None:
        """
        Функция инициализации класса логистической регрессии с возможностью многоклассовой классификации
        :param in_features: Размер входного признакового пространства.
        :param num_classes: Число классов для классификации
        """
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> nn.Linear:
        """
        Прямой проход модели.
        :param x: Входные данные для прямого прохода
        :return: Результаты предсказания модели
        """
        return self.linear(x)

    def evaluate(
            self,
            loader: DataLoader,
    ) -> dict[str, float]:
        """
        Оценивает модель на данных из loader и возвращает значения метрик.
        :param loader: Загрузчик данных
        :return: Словарь со значениями метрик
        """
        self.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.max(outputs, 1)[1]

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = get_metrics_report(all_targets, all_preds)
        metrics['accuracy'] = np.mean(all_targets == all_preds)
        metrics['roc_auc'] = calculate_roc_auc(all_targets, all_probs)

        return metrics

    def evaluate_with_confusion_matrix(
            self,
            loader: DataLoader,
            save_matrix: bool,
            class_names: list[str] = None
    ) -> dict[str, float]:
        """
                Оценивает модель на данных из loader и возвращает значения метрик. Строит матрицу ошибок
                :param loader: Загрузчик данных
                :param save_matrix: Сохранять матрицу ошибок или нет
                :param class_names: Список имён классов
                :return: Словарь со значениями метрик
                """
        self.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = self(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.max(outputs, 1)[1]

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = get_metrics_report(all_targets, all_preds)
        metrics['accuracy'] = np.mean(all_targets == all_preds)
        metrics['roc_auc'] = calculate_roc_auc(all_targets, all_probs)
        if save_matrix:
            plot_confusion_matrix(all_targets, all_preds, class_names, save_path=f'plots/{uuid.uuid4()}/confusion_matrix.png')

        return metrics

    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            optimizer: torch.optim.Optimizer,
            criterion,
            save_matrix: bool = False
    ) -> None:
        """
        Обучает модель с возможномтью ранней остановки.
        :param train_loader: Загрузчик обучающих данных
        :param val_loader: Загрузчик валидационных данных
        :param epochs: Максимальное число эпох
        :param optimizer: Оптимизатор
        :param criterion: Функция потерь
        :param save_matrix: Сохранять матрицу ошибок или нет
        :return: None
        """
        for epoch in range(epochs):
            self.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            val_metrics = self.evaluate(val_loader)
            logging.info(f'Эпоха {epoch + 1}/{epochs} | Точность на Val: {val_metrics["accuracy"]:.4f}')
        val_metrics = self.evaluate_with_confusion_matrix(val_loader, save_matrix)
        logging.info(
            f'Финальные метрики'
            f' | Accuracy: {val_metrics["accuracy"]:.4f}'
            f' | Precision: {val_metrics["precision"]:.4f}'
            f' | Recall: {val_metrics["recall"]:.4f}'
            f' | F1-score: {val_metrics["f1"]:.4f}'
            f' | AUC-ROC: {val_metrics["roc_auc"]:.4f}'
        )


def get_metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Выисляет метрики по переданным данным
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :return: Словарь макроусреднённых значений метрик: accuracy, f1-score, roc-auc, precision, recall
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    eps = 1e-7

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for cls in unique_classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    return {
        'precision': precision_sum / n_classes,
        'recall': recall_sum / n_classes,
        'f1': f1_sum / n_classes
    }


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list[str] = None,
        save_path: str = 'plots/confusion_matrix.png'
) -> None:
    """
    Создаёт и сохраняет график матрицы ошибок
    :param y_true: Истинные метки
    :param y_pred: Предсказанные метки
    :param classes: Имена классов
    :param save_path: Путь для сохранения графика
    :return: None
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def calculate_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Вычисляет ROC-AUC для множественной классификации
    :param y_true: Истинные метки
    :param y_probs: Предсказанные веротяности меток
    :return: Макроусреднённое значение ROC-AUC
    """
    n_classes = y_probs.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    auc_scores = []

    for i in range(n_classes):
        if np.sum(y_true_onehot[:, i]) == 0:
            continue

        try:
            auc = binary_roc_auc(y_true_onehot[:, i], y_probs[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.5)

    return np.mean(auc_scores) if len(auc_scores) != 0 else 0.5


def binary_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Вычисляет ROC-AUC для бинарной классификации
    :param y_true: Истинные метки
    :param y_probs: Предсказанные веротяности меток
    :return: Значение ROC-AUC
    """
    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.5

    sorted_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[sorted_indices]

    tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])

    auc = np.trapz(tpr, fpr)
    return auc
