"""Run minimal gfx906 Triton 3.7 compiler and numerical smoke cases."""

import json

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    values = tl.load(x_ptr + offsets, mask=offsets < size, other=0.0)
    increments = tl.load(y_ptr + offsets, mask=offsets < size, other=0.0)
    tl.store(out_ptr + offsets, values + increments, mask=offsets < size)


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    values = tl.load(x_ptr + row * width + offsets, mask=offsets < width, other=0.0)
    tl.store(out_ptr + row, tl.sum(values, axis=0))


@triton.jit
def matmul_kernel(a_ptr, b_ptr, out_ptr, size: tl.constexpr, block: tl.constexpr):
    offsets_m = tl.arange(0, block)[:, None]
    offsets_n = tl.arange(0, block)[None, :]
    offsets_k = tl.arange(0, block)[None, :]
    a = tl.load(a_ptr + offsets_m * size + offsets_k)
    b = tl.load(b_ptr + offsets_k.T * size + offsets_n)
    tl.store(out_ptr + offsets_m * size + offsets_n, tl.dot(a, b))


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    print(f"{name}=ok")


def main() -> None:
    assert torch.cuda.is_available(), "HIP device is required"
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    assert "gfx906" in name.lower(), f"expected gfx906, got {name!r}"
    assert triton.__version__.startswith("3.7.1"), triton.__version__

    vector_size = 256
    x = torch.randn(vector_size, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    vector_out = torch.empty_like(x)
    vector_add_kernel[(1,)](x, y, vector_out, vector_size, block=256)
    assert_close(vector_out, x + y, "elementwise")

    rows, width = 4, 256
    reduction_input = torch.randn(rows, width, device=device, dtype=torch.float32)
    reduction_out = torch.empty(rows, device=device, dtype=torch.float32)
    row_sum_kernel[(rows,)](reduction_input, reduction_out, width, block=256)
    assert_close(reduction_out, reduction_input.sum(dim=1), "reduction")

    matrix_size = 64
    left = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    right = torch.randn_like(left)
    product = torch.empty_like(left)
    matmul_kernel[(1,)](left, right, product, matrix_size, block=64, num_warps=4)
    assert_close(product.float(), (left @ right).float(), "layout-matmul")

    print(
        json.dumps(
            {
                "device": name,
                "hip": torch.version.hip,
                "torch": torch.__version__,
                "triton": triton.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
