import triton
import triton.language as tl
from utils import make_match_reference, DeterministicContext
import torch
from task import input_t, output_t

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    A, B, out = data
    n_elements = A.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=1024)
    return out

check_implementation = make_match_reference(custom_kernel)