# SlothAi
Performance & Memory Optimization Projects
This repository hosts a collection of advanced projects aimed at optimizing deep learning pipelines. The tasks focus on performance improvements and memory efficiency by leveraging Triton kernels and PyTorch compilation tools. The projects are divided into three distinct challenges:

Convert nf4 to Triton
Enable torch.compile for QLoRA Without Graph Breaks
Memory Efficient Backpropagation
1. Convert nf4 to Triton
Difficulty: Hard
Max Points: 14

Objective
Convert an nf4 quantized tensor into fp16 or bf16 using a single Triton kernel.
Perform the double dequant (handling both the absolute max value and weight formation) within one Triton kernel.
Requirements
Hardware: Must work on a Tesla T4.
Performance: The implementation must be at least 1.15× faster than Unsloth's fast_dequantize.
Memory: Avoid using large intermediate memory buffers.
Constraints:
Do not use torch.compile (though you may use trace.enabled to help in writing Triton kernels).
CUDA is not allowed; however, custom CUDA inside the Triton kernel is permitted.
References & Testing
Good Materials: Review Unsloth's fast_dequantize function and the bitsandbytes dequantize_blockwise.
Testing: Use the provided test_dequantize_function to validate your implementation.

2. Enable torch.compile for QLoRA Without Graph Breaks
Difficulty: Easy to Medium
Max Points: 9

Objective
Develop a single Python script (similar in spirit to task B) that leverages torch.compile to compile all modules where possible.
Ensure that there are no graph breaks and that excessive re-compilations do not occur.
Requirements
Compilation Count: Aim for a maximum of 30 compilations. Exceeding 60 compilations is considered a failure.
Loss Consistency: The loss computed must match the one from the non-compiled module.
Approach:
Leverage patching as extensively as possible.
Identify and disable sections that might cause compilation issues.
Consider regional compilation strategies for efficient code generation.
Performance Logging: Log memory/VRAM usage and monitor speedups.
Compatibility: The solution must work for QLoRA.
3. Memory Efficient Backpropagation
Difficulty: Medium to Hard
Max Points: 10

Background & Objective
In large language models (LLMs), the final layer typically involves a projection 
𝜎
(
𝑋
𝑊
)
σ(XW) to compute the next-token probabilities. For very large vocabulary sizes (e.g., 128K), materializing the logits can cause significant VRAM spikes. For instance:

Example:
Batch size (bsz) = 4
Sequence length (qlen) = 4096
Hidden dimension (hd) = 4096
Vocabulary size (vocab) = 128K
This setup can require up to 4GB (or 8GB if upcast to float32) of VRAM solely for the logits.
The goal is to avoid materializing these large intermediate tensors by computing them on the fly.

Implementation Strategy
On-the-Fly Computation:
Instead of materializing the intermediate logits, compute the transformation function's output directly. For example, if intermediate logits for two batches are normally gathered as:

[
𝑥
1
 
𝑥
2
]
×
𝑊
𝑓
  
(
[
𝑥
1
𝑊
 
𝑥
2
𝑊
]
)
=
[
𝑥
1
𝑊
 
𝑥
2
𝑊
]
=
(
𝑦
1
 
𝑦
2
)
[x 
1
​
 x 
2
​
 ]×W 
f
​
 ([x 
1
​
 Wx 
2
​
 W])=[x 
1
​
 Wx 
2
​
 W]=(y 
1
​
 y 
2
​
 )
then processing batches sequentially can significantly reduce VRAM usage (e.g., from 4GB to 2GB).

Backpropagation Considerations:
Use the chain rule during backpropagation. Instead of hard-coding derivatives, implement a custom torch.autograd.Function with a forward and backward pass that computes the intermediate tensors on the fly. This approach:

Moves the transformation from large intermediate tensors to a smaller tensor suitable for autograd.
Ensures proper propagation of upstream gradients through multiplication with the computed gradients.
Hints & References:

Refer to torch.checkpoint for ideas on managing intermediate activations and gradient flow.
Ensure that the Cross Entropy Loss (and other functions) work seamlessly with your implementation.
Testing & Validation
Unit Tests:
Use the provided test functions (e.g., test_dequantize_function) to validate both correctness and performance.
Performance Metrics:
For the nf4 to Triton project, confirm that your implementation exceeds Unsloth's version by at least 1.15×.
For the QLoRA task, ensure that the total compilations remain below the specified threshold and that memory/VRAM usage is monitored appropriately.
For the memory-efficient backprop, verify that the loss remains consistent with the non-optimized implementation while reducing VRAM usage.
Contributing
Contributions, suggestions, and improvements are welcome. Please ensure that:

All changes maintain or improve the performance/memory efficiency benchmarks.
Local tests pass before submitting a pull request.
Detailed comments and documentation are provided for complex kernel operations or custom autograd functions.
License
This project is licensed under the Apache License 2.0.
For more details, please refer to the LICENSE file.
