import os
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from homework_model_modification import *


def make_regression_data(n_samples=100, n_features=3):
    X = torch.randn(n_samples, n_features)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y


def make_classification_data():
    X_pos = torch.randn(50, 2) + 5
    X_neg = torch.randn(50, 2) - 5
    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([torch.ones(50, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    return X, y


class DummyDataset:
    def __init__(self, X, y, is_regression=True):
        self.X = X
        self.y = y
        self.is_regression = is_regression

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.is_regression:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]


@pytest.fixture
def regression_loader():
    X, y = make_regression_data()
    ds = DummyDataset(X, y, is_regression=True)
    return DataLoader(ds, batch_size=20)


@pytest.fixture
def classification_loader():
    X, y = make_classification_data()
    ds = DummyDataset(X, y, is_regression=False)
    return DataLoader(ds, batch_size=20)


def test_linear_forward_and_reg_loss():
    model = LinearRegressionWithRegularization(in_features=3, l1_lambda=0.5, regularization='l1')
    x = torch.tensor([[1.0, -1.0, 2.0]])
    out = model(x)
    assert out.shape == (1, 1)
    reg_loss = model.regularization_loss()
    assert reg_loss.item() >= 0


def test_linear_train_and_evaluate(regression_loader):
    model = LinearRegressionWithRegularization(in_features=3, regularization='l2', l2_lambda=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    model.train_model(
        train_loader=regression_loader,
        val_loader=regression_loader,
        epochs=3,
        optimizer=optimizer,
        criterion=criterion,
        patience=2
    )
    val_loss = model.evaluate(regression_loader, criterion)
    assert isinstance(val_loss, float)
    assert val_loss >= 0


def test_logistic_evaluate_and_metrics(classification_loader):
    model = LogisticRegressionMultiClass(in_features=2, num_classes=2)
    metrics = model.evaluate(classification_loader)
    assert 'accuracy' in metrics and 'f1' in metrics and 'roc_auc' in metrics
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_metrics_report():
    y_true = np.array([0, 1, 1, 0, 2, 2])
    y_pred = y_true.copy()
    rep = get_metrics_report(y_true, y_pred)
    assert pytest.approx(rep['precision'], rel=1e-3) == 1.0
    assert pytest.approx(rep['recall'], rel=1e-3) == 1.0
    assert pytest.approx(rep['f1'], rel=1e-3) == 1.0


def test_binary_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])
    auc = binary_roc_auc(y_true, y_probs)
    assert pytest.approx(auc, rel=1e-3) == 1.0


def test_calculate_roc_auc_multiclass():
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_probs = np.eye(3)[y_true]
    auc = calculate_roc_auc(y_true, y_probs)
    assert pytest.approx(auc, rel=1e-3) == 1.0


def test_plot_confusion_matrix(tmp_path):
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    save_path = tmp_path / 'cm.png'
    plot_confusion_matrix(y_true, y_pred, classes=['A', 'B'], save_path=str(save_path))
    assert save_path.exists()
