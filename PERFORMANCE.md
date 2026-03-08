## 🚀 Training Performance Profile (PyTorch 2.x)

This section documents the computational efficiency of the **Yiddish MAE-ViT** training script on a Multi-GPU setup.

### 💻 Hardware Configuration

* **GPU:** 2x NVIDIA GeForce RTX 3090 (Ampere Architecture)
* **VRAM:** 24 GB per card (Total 48 GB)
* **Compute Capability:** 8.6

### 📊 Profiler Summary (Step Breakdown)

The following metrics represent the average execution time during a single training step.

| Category | Time Duration (us) | Percentage (%) |
| --- | --- | --- |
| **Average Step Time** | **330,525** | **100.00%** |
| **Kernel (GPU Execution)** | 325,776 | 98.56% |
| **Memcpy (HtoD/DtoH)** | 2,541 | 0.77% |
| **CPU Execution** | 1,681 | 0.51% |
| **DataLoader (Wait time)** | 0 | 0.00% |
| **Communication / Runtime** | 30 | 0.01% |
| **Other** | 480 | 0.15% |

### 📈 Resource Utilization Details (GPU 0)

Detailed hardware telemetry during the active profiling window:

* **GPU Utilization:** `98.92 %` (Near-perfect saturation)
* **Est. SM Efficiency:** `93.07 %`
* **Est. Achieved Occupancy:** `23.78 %`
* **Memory Used:** `23.59 GB` / `24.00 GB`

### 🔍 Performance Analysis

1. **Zero Data Bottleneck:** The `DataLoader` overhead is at **0.00%**, indicating that the CPU (with `num_workers=12` and `pin_memory=True`) is successfully pre-fetching batches before the GPU requests them.
2. **Compute Bound:** Over **98%** of the step time is spent on actual GPU kernels. The script is heavily compute-bound, which is the ideal state for deep learning training.
3. **Occupancy Notes:** While GPU Utilization is high, the *Achieved Occupancy* (23.78%) suggests there is still room for more instruction-level parallelism. Further speedups could be achieved by utilizing `torch.compile()` to fuse kernels and reduce memory-bound overheads in the Attention layers.

### 🛠️ Optimization Strategies Applied

To achieve the current performance metrics, the following optimizations were implemented:

#### 1. **Mixed Precision Training (`bf16`)**

* **Strategy:** Utilized `accelerate` with `mixed_precision="bf16"`.
* **Impact:** Leveraged NVIDIA Ampere Tensor Cores for faster matrix multiplications while maintaining numerical stability (critical for ViT architectures). This reduced VRAM footprint, allowing for a larger batch size.

#### 2. **Advanced Data Loading Pipeline**

* **`num_workers=12`**: Parallelized data augmentation and loading across CPU cores.
* **`pin_memory=True`**: Enabled faster Tensors transfer from host (RAM) to device (VRAM).
* **`persistent_workers=True`**: Prevented the overhead of re-initializing worker processes between training epochs.
* **`prefetch_factor=2`**: Ensured that the GPU always has 2 batches ready in the queue, eliminating the **0.00% DataLoader wait time** observed in the profile.

#### 3. **Memory & Compute Optimization**

* **`set_to_none=True`**: Optimized `optimizer.zero_grad()` by setting gradients to `None` instead of zeroing them, reducing memory bandwidth pressure.
* **Distributed Data Parallel (DDP)**: Used `accelerate` to handle multi-GPU communication efficiently via the NCCL backend.

#### 4. **Future Performance Roadmap**

* [ ] **Kernel Fusion:** Implementation of `torch.compile(model)` to increase *Achieved Occupancy* (currently 23.78%) by fusing pointwise and reduction operations.
* [ ] **TF32 Activation:** Setting `torch.set_float32_matmul_precision('high')` to further accelerate `float32` operations if they occur.
* [ ] **FlashAttention:** Investigating `torch.nn.functional.scaled_dot_product_attention` for more memory-efficient attention mechanisms.


torch.compile reduced epoch pass time by ~1s to 7.5s per epoch