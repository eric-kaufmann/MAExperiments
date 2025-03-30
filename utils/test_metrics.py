import torch
from metrics import cosine_similarity, mse_error, mae_error

def test_cosine_similarity():
    pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = cosine_similarity(pred, target)
    expected = torch.tensor(1.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

    pred = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    result = cosine_similarity(pred, target)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_mse_error():
    pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = mse_error(pred, target)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

    pred = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    result = mse_error(pred, target)
    expected = torch.tensor(2/3)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

def test_mae_error():
    pred = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    result = mae_error(pred, target)
    expected = torch.tensor(0.0)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

    pred = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    target = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    result = mae_error(pred, target)
    expected = torch.tensor(2/3)
    assert torch.isclose(result, expected), f"Expected {expected}, but got {result}"

if __name__ == "__main__":
    test_cosine_similarity()
    test_mse_error()
    test_mae_error()
    print("All tests passed!")