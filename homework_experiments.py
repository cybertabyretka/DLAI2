import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures

from homework_datasets import CSVDataset
from homework_model_modification import LinearRegressionWithRegularization, LogisticRegressionMultiClass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_results(results: dict[tuple, float], model_type: str) -> None:
    """
    Визуализация результатов экспериментов с логарифмической шкалой и фильтрацией выбросов
    :param results: Метрики, полученные при обучении
    :param model_type: Тип модели
    :return: None
    """
    plt.figure(figsize=(12, 6))
    filtered_values = [v if np.isfinite(v) and v < 1e3 else 1e3 for v in results.values()]
    labels = [f"LR={lr}\nBS={bs}\nOpt={opt.__name__}" for (lr, bs, opt) in results.keys()]
    plt.bar(range(len(filtered_values)), filtered_values, tick_label=labels)
    plt.title('Hyperparameter Tuning Results')
    plt.ylabel('MSE Loss' if model_type == 'regression' else 'F1 Score')
    plt.yscale('log')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'plots/hyperparameter_results_{model_type}.png')
    plt.close()


def run_hyperparameter_experiments(dataset: CSVDataset, model_type: str = 'regression') -> dict:
    """
    Выполняет эксперимент с подбором гиперпараметров
    :param dataset: Датасет, на котором проводится эксперимент
    :param model_type: Тип модели (regression или другое)
    :return: Словарь метрик при различных значениях гиперпараметров
    """
    results = {}
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    optimizers = [optim.SGD, optim.Adam, optim.RMSprop]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for opt in optimizers:
                logging.info(f"Текущий эксперимент: lr={lr}, bs={batch_size}, opt={opt}")

                train_loader, val_loader = dataset.get_dataloaders(batch_size=batch_size)
                num_features = len(dataset.data.columns) - 1

                if model_type == 'regression':
                    model = LinearRegressionWithRegularization(num_features)
                else:
                    num_classes = dataset.data[dataset.target_column].nunique()
                    model = LogisticRegressionMultiClass(num_features, num_classes=num_classes)

                optimizer = opt(model.parameters(), lr=lr)

                if model_type == 'regression':
                    criterion = nn.MSELoss()
                    model.train_model(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=100,
                        optimizer=optimizer,
                        criterion=criterion
                    )
                    loss = model.evaluate(val_loader, criterion)
                    results[(lr, batch_size, opt)] = loss
                else:
                    criterion = nn.CrossEntropyLoss()
                    model.train_model(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=100,
                        optimizer=optimizer,
                        criterion=criterion
                    )
                    metrics = model.evaluate(val_loader)
                    results[(lr, batch_size, opt)] = metrics['f1']

    visualize_results(results, model_type)
    return results


def experiment_polynomial(dataset: CSVDataset) -> float:
    """
    Экспериментирует с полиномиальными признаками degree=2
    :param dataset: Датасет, на котором проводится эксперимент
    :return: Значение метрики
    """
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(dataset.data.drop(columns=[dataset.target_column]))
    cols_poly = [f"poly_{i}" for i in range(X_poly.shape[1])]
    df_poly = pd.DataFrame(X_poly, columns=cols_poly)
    df_poly[dataset.target_column] = dataset.data[dataset.target_column].values
    poly_ds = CSVDataset(
        data_frame=df_poly,
        target_column=dataset.target_column,
        num_features=cols_poly,
        is_regression=True
    )
    model = LinearRegressionWithRegularization(len(cols_poly))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train_model(*poly_ds.get_dataloaders(batch_size=64), epochs=100, optimizer=optimizer, criterion=criterion)
    return model.evaluate(poly_ds.get_dataloaders()[1], criterion=criterion)


def experiment_interactions(dataset: CSVDataset) -> float:
    """
    Экспериментирует с interaction_only
    :param dataset: Датасет, на котором проводится эксперимент
    :return: Значение метрики
    """
    inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_int = inter.fit_transform(dataset.data.drop(columns=[dataset.target_column]))
    cols_int = [f"int_{i}" for i in range(X_int.shape[1])]
    df_int = pd.DataFrame(X_int, columns=cols_int)
    df_int[dataset.target_column] = dataset.data[dataset.target_column].values
    int_ds = CSVDataset(
        data_frame=df_int,
        target_column=dataset.target_column,
        num_features=cols_int,
        is_regression=True
    )
    model = LinearRegressionWithRegularization(len(cols_int))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train_model(*int_ds.get_dataloaders(batch_size=64), epochs=100, optimizer=optimizer, criterion=criterion)
    return model.evaluate(int_ds.get_dataloaders()[1], criterion=criterion)


def experiment_statistics(dataset: CSVDataset) -> float:
    """
    Экспериментирует со статистическими признаками: среднее и дисперсия по строке
    :param dataset: Датасет, на котором проводится эксперимент
    :return: Значение метрики
    """
    # Статистические признаки: среднее и дисперсия по строке
    X = dataset.data.drop(columns=[dataset.target_column]).values
    row_mean = X.mean(axis=1).reshape(-1,1)
    row_var = X.var(axis=1).reshape(-1,1)
    df_stat = pd.DataFrame(np.hstack([row_mean, row_var]), columns=["mean", "var"])
    df_stat[dataset.target_column] = dataset.data[dataset.target_column].values
    stat_ds = CSVDataset(
        data_frame=df_stat,
        target_column=dataset.target_column,
        num_features=["mean", "var"],
        is_regression=True
    )
    model = LinearRegressionWithRegularization(2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train_model(*stat_ds.get_dataloaders(batch_size=64), epochs=100, optimizer=optimizer, criterion=criterion)
    return model.evaluate(stat_ds.get_dataloaders()[1], criterion=criterion)


if __name__ == "__main__":
    print(f'{"-" * 70}\n3.1 Исследование гиперпараметров\n{"-" * 70}')
    # 3.1 Исследование гиперпараметров
    regression_dataset = CSVDataset(
        'data/possum_fixed.csv',
        target_column='hdlngth',
        num_features=['site', 'skullw', 'totlngth', 'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly'],
        cat_features=['age'],
        binary_features=['Victoria', 'male']
    )
    run_hyperparameter_experiments(regression_dataset, 'regression')

    classification_dataset = CSVDataset(
        'data/glass.csv',
        target_column='Type',
        num_features=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],
        cat_features=[],
        binary_features=[],
        is_regression=False
    )
    run_hyperparameter_experiments(classification_dataset, 'classification')

    print(f'{"-" * 70}\n3.2 Feature Engineering\n{"-" * 70}')
    # 3.2 Feature Engineering
    # Эксперимент 1: Полиномиальные признаки
    print(f'{"-" * 70}\nЭксперимент 1: Полиномиальные признаки\n{"-" * 70}')
    mse_poly = experiment_polynomial(regression_dataset)
    print(f"MSE (polynomial): {mse_poly:.4f}")

    # Эксперимент 2: Взаимодействия
    print(f'{"-" * 70}\nЭксперимент 2: Взаимодействия\n{"-" * 70}')
    mse_int = experiment_interactions(regression_dataset)
    print(f"MSE (interactions): {mse_int:.4f}")

    # Эксперимент 3: Статистические признаки
    print(f'{"-" * 70}\nЭксперимент 3: Статистические признаки\n{"-" * 70}')
    mse_stat = experiment_statistics(regression_dataset)
    print(f"MSE (statistics): {mse_stat:.4f}")

    # Сравнение
    print(f'{"-" * 70}\nСравнение\n{"-" * 70}')
    base_mse = LinearRegressionWithRegularization(len(regression_dataset.data.columns) - 1)
    base_mse.train_model(*regression_dataset.get_dataloaders(batch_size=64), epochs=100,
                         optimizer=optim.Adam(base_mse.parameters(), lr=0.01), criterion=nn.MSELoss())
    mse_base = base_mse.evaluate(regression_dataset.get_dataloaders()[1], criterion=nn.MSELoss())
    print(f"MSE (base): {mse_base:.4f}")
    print(f"Улучшение poly vs base: {(mse_base - mse_poly) / mse_base * 100:.2f}%")
    print(f"Улучшение int vs base: {(mse_base - mse_int) / mse_base * 100:.2f}%")
    print(f"Улучшение stat vs base: {(mse_base - mse_stat) / mse_base * 100:.2f}%")
