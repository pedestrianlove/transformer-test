import torch


def test_tensor_operations():
    __x__ = torch.tensor([1, 2, 3])
    __y__ = torch.tensor([9, 8, 7])

    # Addition
    assert (__x__ + __y__ == torch.tensor([10, 10, 10])).all()
    assert (__x__ + __y__ == torch.add(__x__, __y__)).all()

    # Subtraction
    assert (__x__ - __y__ == torch.tensor([-8, -6, -4])).all()
    assert (__x__ - __y__ == torch.sub(__x__, __y__)).all()

    # Multiplication
    assert (__x__ * __y__ == torch.tensor([9, 16, 21])).all()
    assert (__x__ * __y__ == torch.mul(__x__, __y__)).all()

    # Division
    assert (__x__ / __y__ == torch.tensor([1 / 9, 2 / 8, 3 / 7])).all()
    assert (__x__ / __y__ == torch.div(__x__, __y__)).all()

    # Power
    assert (__x__**2 == torch.tensor([1, 4, 9])).all()
    assert (__x__**2 == torch.pow(__x__, 2)).all()


def test_tensor_matmul():
    __x__ = torch.tensor([1, 2, 3])
    __y__ = torch.tensor([9, 8, 7])
    __z__ = torch.rand((3, 1))

    # Test matmul
    assert __z__.mm(__x__.reshape(1, 3).float()).shape == (3, 3)
    assert (
        __z__.mm(__x__.reshape(1, 3).float())
        == torch.mm(__z__, __x__.reshape(1, 3).float())
    ).all()

    # Test mat power
    __A__ = torch.tensor([[1, 2], [3, 4]])
    assert __A__.matrix_power(3).shape == (2, 2)
    assert (__A__.matrix_power(3) == torch.tensor([[37, 54], [81, 118]])).all()

    # Test dot product
    assert torch.dot(__x__, __y__) == 1 * 9 + 2 * 8 + 3 * 7
    assert torch.dot(__x__, __y__) == torch.sum(__x__ * __y__)
