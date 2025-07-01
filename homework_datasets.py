import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader

from homework_model_modification import LinearRegressionWithRegularization, LogisticRegressionMultiClass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 2.1 Кастомный Dataset класс
class CSVDataset(Dataset):
    """
    Класс для обработки датасета в формате csv.
    """
    def __init__(
            self,
            file_path: str = None,
            data_frame: pd.DataFrame = None,
            target_column: str = None,
            num_features: list[str] = None,
            cat_features: list[str] = None,
            binary_features: list[str] = None,
            is_regression: bool = True
    ) -> None:
        """
        Функция инициализации класса для обработки датасета в формате csv.
        :param file_path: Путь до файла с датасетом (исключающийся с data_frame)
        :param data_frame: Dataframe с содержащимся в нём датасете (исключающийся с file_path)
        :param target_column: Имя целевой колонки
        :param num_features: Имена численных (вещественных или целочисленных) колонок
        :param cat_features: Имена категориальных колонок
        :param binary_features: Имена бинарных колонок
        :param is_regression: Для какой задачи датасет (регрессия или другое)
        """
        if file_path is not None:
            self.data = pd.read_csv(file_path)
        elif data_frame is not None:
            self.data = data_frame
        else:
            raise ValueError("Либо file_path, либо data_frame должен быть передан")

        self.target_column = target_column
        self.num_features = num_features or []
        self.cat_features = cat_features or []
        self.binary_features = binary_features or []
        self.is_regression = is_regression

        if file_path is not None:
            self.preprocess_data()
        logging.info(f"Dataset loaded: {len(self.data)} samples")

    def __len__(self) -> int:
        """
        Возвращает количество записей в датасете
        :return: Количество записей в датасете
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает отдельный объект из датасета
        :param idx: Индекс, объект по которому нужно получить
        :return: Кортеж из признаков и таргета
        """
        features = self.data.drop(columns=[self.target_column]).iloc[idx].values
        target = self.data[self.target_column].iloc[idx]

        if self.is_regression:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor([target], dtype=torch.float32)
            )
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long)
        )

    def preprocess_data(self) -> None:
        """
        Выполняет предобработку датасета. Обрабатывает числовые, категориальные и бинарные признаки
        :return: None
        """
        if self.num_features:
            scaler = StandardScaler()
            self.data[self.num_features] = scaler.fit_transform(self.data[self.num_features])

        for feature in self.cat_features:
            encoder = LabelEncoder()
            self.data[feature] = encoder.fit_transform(self.data[feature])

        for feature in self.binary_features:
            self.data[feature] = self.data[feature].astype(int)

        if not self.is_regression:
            le = LabelEncoder()
            self.data[self.target_column] = le.fit_transform(self.data[self.target_column])

    def get_dataloaders(
            self,
            test_size: float = 0.2,
            batch_size: int = 32
    ) -> tuple[DataLoader, DataLoader]:
        """
        Возвращает два dataloader`а. Один для обучения, второй для тестирования.
        :param test_size: Размер выборки для тестирования. По умолчанию 0.2
        :param batch_size: Размер пакета для единовременной обработки. По умолчанию 32
        :return: Два dataloader`а. Один для обучения, второй для тестирования.
        """
        train_df, test_df = train_test_split(self.data, test_size=test_size)

        train_dataset = CSVDataset(
            data_frame=train_df,
            target_column=self.target_column,
            num_features=self.num_features,
            cat_features=self.cat_features,
            binary_features=self.binary_features,
            is_regression=self.is_regression
        )

        test_dataset = CSVDataset(
            data_frame=test_df,
            target_column=self.target_column,
            num_features=self.num_features,
            cat_features=self.cat_features,
            binary_features=self.binary_features,
            is_regression=self.is_regression
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader


# 2.2 Эксперименты с различными датасетами
if __name__ == '__main__':
    # Проверка регрессии
    print(f'{"-"*70}\nПроверка регрессии\n{"-"*70}')
    regression_dataset = CSVDataset(
        'data/possum_fixed.csv',
        target_column='hdlngth',
        num_features=['site', 'skullw', 'totlngth', 'taill', 'footlgth', 'earconch', 'eye', 'chest', 'belly'],
        cat_features=['age'],
        binary_features=['Victoria', 'male']
    )
    num_features = len(regression_dataset.data.columns) - 1
    linear_model = LinearRegressionWithRegularization(num_features)
    train_loader, val_loader = regression_dataset.get_dataloaders(batch_size=128)
    optimizer = optim.Adam(linear_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    linear_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        optimizer=optimizer,
        criterion=criterion
    )
    torch.save(linear_model.state_dict(), 'models/linear_model.pth')

    # Проверка классификации
    print(f'{"-" * 70}\nПроверка классификации\n{"-" * 70}')
    classification_dataset = CSVDataset(
        'data/glass.csv',
        target_column='Type',
        num_features=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'],
        cat_features=[],
        binary_features=[],
        is_regression=False
    )
    num_features = len(classification_dataset.data.columns) - 1
    logistic_model = LogisticRegressionMultiClass(num_features, classification_dataset.data['Type'].nunique())
    train_loader, val_loader = classification_dataset.get_dataloaders(batch_size=128)
    optimizer = optim.Adam(logistic_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    logistic_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        optimizer=optimizer,
        criterion=criterion,
        save_matrix=True
    )
    torch.save(logistic_model.state_dict(), 'models/logistic_model.pth')
