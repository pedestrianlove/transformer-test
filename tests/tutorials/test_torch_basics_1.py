import torch


def test_tensor_behavior():
    __tensor__ = torch.tensor(
        [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True
    )

    # Check the shape, dtype, requires_grad attribute of the tensor
    assert __tensor__.shape == (2, 3), (
        f"Expected shape (2, 3), but got {__tensor__.shape}"
    )
    assert __tensor__.dtype == torch.float32, (
        f"Expected dtype torch.float32, but got {__tensor__.dtype}"
    )
    assert __tensor__.requires_grad, (
        f"Expected requires_grad to be True, but got {__tensor__.requires_grad}"
    )


def test_tensor_type_conversion():
    __tensor__ = torch.arange(4)

    # Convert the tensor to a different dtype and check the new dtype
    # Test for short/long/half/float/double
    assert __tensor__.dtype == torch.int64, (
        f"Expected dtype torch.int64, but got {__tensor__.dtype}"
    )
    assert __tensor__.short().dtype == torch.int16, (
        f"Expected dtype torch.int16, but got {__tensor__.short().dtype}"
    )
    assert __tensor__.long().dtype == torch.int64, (
        f"Expected dtype torch.int64, but got {__tensor__.long().dtype}"
    )
    assert __tensor__.half().dtype == torch.float16, (
        f"Expected dtype torch.float16, but got {__tensor__.half().dtype}"
    )
    assert __tensor__.float().dtype == torch.float32, (
        f"Expected dtype torch.float32, but got {__tensor__.float().dtype}"
    )
