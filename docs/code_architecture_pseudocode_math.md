# Code Architecture, Pseudocode, and Mathematical Background

**Project**: HPC Performance Analysis of CNN Training — GPU vs Multi-threaded CPU
**Date**: 2026-03-29

---

## Table of Contents

1. [Overall System Architecture](#1-overall-system-architecture)
2. [CNN Model Architectures](#2-cnn-model-architectures)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation and Metrics](#5-evaluation-and-metrics)
6. [Custom CUDA Kernel Architecture](#6-custom-cuda-kernel-architecture)
7. [HPC Parallelization Mathematics](#7-hpc-parallelization-mathematics)

---

## 1. Overall System Architecture

### 1.1 Experiment Structure

Five experiments systematically evaluate CNN training performance across hardware configurations:

| Experiment | Platform | Purpose | Scaling Variable |
|---|---|---|---|
| A | CPU (Barbun, 1 thread) | Sequential baseline | None (reference) |
| B | GPU (akya-cuda, V100) | Standard PyTorch CUDA | CPU-side threads |
| C | GPU (akya-cuda, V100) | Custom CUDA kernels | CUDA thread geometry |
| D | GPU (akya-cuda, V100) | Batch size analysis | Batch size (1-256) |
| E | CPU (Hamsi, 2x28 cores) | Multi-thread scaling | Thread count (1-112) |

All experiments share: VGG-19 + ResNet-50 models, 3-class dataset, 192x192 images.

> Architecture diagrams are provided as JPEG images in `paper_figures/arch_*.jpg`

### 1.2 Software Stack

```
APPLICATION LAYER
  train_cpu.py | train_cuda.py | train_custom.py
  train_batch_scaling.py | train_cpu_parallel.py

FRAMEWORK LAYER
  PyTorch (nn, optim, autograd) | torchvision (models, transforms) | sklearn (metrics)

COMPUTE BACKENDS
  CPU: MKL/OpenMP, ATen Threads, BLAS/LAPACK
  GPU: cuDNN, cuBLAS
  Custom: custom_ops CUDA extension

HARDWARE LAYER
  CPU: Intel Xeon Gold 6258R (2x28 cores, NUMA)
  GPU: NVIDIA V100 (32 GB HBM2, 5120 CUDA cores, 80 SMs)
```

### 1.3 End-to-End Data Flow

```
Raw Images (3000) --> Augmentation + Normalize (192x192) --> Stratified Split (70/15/15)
    --> DataLoader (Batched) --> Training Loop --> Evaluation
```

Training loop per epoch:
```
Forward Pass --> Loss (Weighted CE) --> Backward Pass (Gradients) --> Optimizer (Adam)
    --> Validation --> LR Scheduler --> Best Model Checkpoint
```

---

## 2. CNN Model Architectures

### 2.1 VGG-19 Architecture

VGG-19 (Visual Geometry Group, 2014) — Configuration E. 19 weighted layers:
16 convolutional + 3 fully connected.

**Parameters**: ~143.7M (for 3 classes)

```
INPUT: [N, 3, 192, 192]

BLOCK 1: Conv3-64 x2 + BN + ReLU + MaxPool    --> [N, 64, 96, 96]
BLOCK 2: Conv3-128 x2 + BN + ReLU + MaxPool   --> [N, 128, 48, 48]
BLOCK 3: Conv3-256 x4 + BN + ReLU + MaxPool   --> [N, 256, 24, 24]
BLOCK 4: Conv3-512 x4 + BN + ReLU + MaxPool   --> [N, 512, 12, 12]
BLOCK 5: Conv3-512 x4 + BN + ReLU + MaxPool   --> [N, 512, 6, 6]

AdaptiveAvgPool2d(1x1)                         --> [N, 512]
FC(512, 4096) + ReLU + Dropout(0.5)            --> [N, 4096]
FC(4096, 4096) + ReLU + Dropout(0.5)           --> [N, 4096]
FC(4096, 3)                                    --> [N, 3]

OUTPUT: [N, 3] (logits: all_snow, no_snow, partial)
```

**Parameter Distribution**:

| Layer Group | Parameters | Percentage |
|---|---|---|
| Conv Blocks (16 layers) | ~20.0M | 13.9% |
| FC1 (512 -> 4096) | ~2.1M | 1.5% |
| FC2 (4096 -> 4096) | ~16.8M | 11.7% |
| FC3 (4096 -> 3) | ~12K | <0.1% |
| BatchNorm (16 layers) | ~10K | <0.1% |
| **Total** | **~143.7M** | **100%** |

> **Note**: Standard VGG-19 has FC1 input of 512x7x7=25088. In this study,
> AdaptiveAvgPool2d(1) reduces FC1 input to 512, significantly reducing parameter count.

### 2.2 ResNet-50 Architecture

ResNet-50 (Residual Network, 2015) — bottleneck blocks with skip connections.
50 weighted layers.

**Parameters**: ~23.5M (for 3 classes)

```
INPUT: [N, 3, 192, 192]

STEM:    Conv7x7-64 (stride=2) + BN + ReLU     --> [N, 64, 96, 96]
         MaxPool3x3 (stride=2, pad=1)           --> [N, 64, 48, 48]

STAGE 1: 3 Bottleneck blocks (64->256)          --> [N, 256, 48, 48]
STAGE 2: 4 Bottleneck blocks (128->512)         --> [N, 512, 24, 24]
STAGE 3: 6 Bottleneck blocks (256->1024)        --> [N, 1024, 12, 12]
STAGE 4: 3 Bottleneck blocks (512->2048)        --> [N, 2048, 6, 6]

AdaptiveAvgPool2d(1x1)                         --> [N, 2048]
FC(2048, 3)                                    --> [N, 3]

OUTPUT: [N, 3] (logits)
```

**Bottleneck Block Detail (Expansion = 4)**:

```
Input x [C_in, H, W]
    |
    +--> 1x1 Conv (reduce): C_in -> C_mid, BN, ReLU
    |    3x3 Conv (spatial): C_mid -> C_mid, stride=s, pad=1, BN, ReLU
    |    1x1 Conv (expand):  C_mid -> C_mid*4, BN (no ReLU!)
    |                               |
    +--- identity (or downsample) --+
                                    |
                                   (+) element-wise addition
                                    |
                                   ReLU
                                    |
Output [C_mid*4, H/s, W/s]
```

**Stage Details**:

| Stage | Blocks | Input Size | Output Size | C_mid |
|---|---|---|---|---|
| Stem | - | 3x192x192 | 64x48x48 | - |
| Layer1 | 3 | 64x48x48 | 256x48x48 | 64 |
| Layer2 | 4 | 256x48x48 | 512x24x24 | 128 |
| Layer3 | 6 | 512x24x24 | 1024x12x12 | 256 |
| Layer4 | 3 | 1024x12x12 | 2048x6x6 | 512 |

### 2.3 VGG-19 vs ResNet-50 Architectural Comparison

| Aspect | VGG-19 | ResNet-50 |
|---|---|---|
| Topology | Sequential chain | Residual (skip connections) |
| Parameters | 143.7M | 23.5M |
| FLOPs (192x192) | ~19.6 GFLOPS | ~4.1 GFLOPS |
| Dominant ops | Large FC layers (BLAS-heavy) | Many small conv ops |
| Serial fraction (f) | ~1.8% (scales well) | ~63.4% (scales poorly) |
| Gradient flow | Prone to vanishing gradients | Skip connections preserve gradients |

---

## 3. Data Processing Pipeline

### 3.1 Dataset Structure

```
Data/
  all_snow/     749 images (25.0%)  -- ~7.7x augmented from 97 originals
  no_snow/     2027 images (67.6%)  -- ~7.5x augmented from 270 originals
  partial/      224 images (7.5%)   -- ~8.3x augmented from 27 originals
                -----
Total:         3000 images (192x192 JPEG)
```

### 3.2 Data Augmentation Pseudocode

```
ALGORITHM: DataAugmentation
-------------------------------------------------
Input:  raw_image (variable size, RGB)
Output: augmented_tensor [3, 192, 192]

1.  img <- Resize(raw_image, 192x192)

2.  IF mode == TRAIN:
        IF random() < 0.5:
            img <- HorizontalFlip(img)

        theta <- Uniform(-10, +10) degrees
        img <- Rotate(img, theta)

        delta_brightness <- Uniform(0.8, 1.2)
        delta_contrast   <- Uniform(0.8, 1.2)
        img <- ColorJitter(img, delta_brightness, delta_contrast)

3.  tensor <- ToTensor(img)    // [H,W,3] -> [3,H,W], [0,255] -> [0,1]

4.  FOR c = 0, 1, 2:
        tensor[c] <- (tensor[c] - mu[c]) / sigma[c]
    WHERE mu    = [0.485, 0.456, 0.406]   (ImageNet means)
          sigma = [0.229, 0.224, 0.225]   (ImageNet stds)

5.  RETURN tensor
```

**Mathematical Expression -- Normalization**:

x_hat(c,h,w) = (x(c,h,w) - mu_c) / sigma_c

### 3.3 Stratified Split

```
ALGORITHM: StratifiedSplit
-------------------------------------------------
Input:  dataset (N images, K classes), ratios (0.70, 0.15, 0.15)
Output: train_idx, val_idx, test_idx

1.  targets <- all class labels

2.  FOR cls = 0 TO K-1:
        cls_indices <- indices where targets == cls
        Shuffle(cls_indices, seed=42)

        n <- |cls_indices|
        n_train <- floor(n * 0.70)
        n_val   <- floor(n * 0.15)

        train_idx <- train_idx U cls_indices[0 : n_train]
        val_idx   <- val_idx   U cls_indices[n_train : n_train+n_val]
        test_idx  <- test_idx  U cls_indices[n_train+n_val : n]

3.  RETURN train_idx, val_idx, test_idx
```

**Resulting Split**:

| Class | Total | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|
| all_snow | 749 | 524 | 112 | 113 |
| no_snow | 2027 | 1418 | 304 | 305 |
| partial | 224 | 156 | 33 | 35 |
| **Total** | **3000** | **2098** | **449** | **453** |

### 3.4 Class Weighting (Inverse Frequency)

```
ALGORITHM: ComputeClassWeights
-------------------------------------------------
Input:  train_targets (training set labels)
Output: class_weights [K]

1.  FOR cls = 0 TO K-1:
        count[cls] <- number of cls in train_targets

2.  FOR cls = 0 TO K-1:
        w[cls] <- 1.0 / count[cls]

3.  w <- w / sum(w) * K     // normalize so sum = K

4.  RETURN w
```

**Mathematical Expression**:

w_c = K / (sum_{j=0}^{K-1} 1/n_j) * 1/n_c

Where n_c is the number of training samples for class c, K is the number of classes.

---

## 4. Training Pipeline

### 4.1 Forward Pass

```
ALGORITHM: ForwardPass
-------------------------------------------------
Input:  x [N, 3, 192, 192] (batch input), model
Output: logits [N, K]

// VGG-19:
1.  FOR each (Conv, BN, ReLU, Pool) block:
        x <- Pool(ReLU(BN(Conv(x))))
2.  x <- AdaptiveAvgPool2d(x)           // [N, 512, 1, 1]
3.  x <- Flatten(x)                      // [N, 512]
4.  x <- Dropout(ReLU(FC1(x)))           // [N, 4096]
5.  x <- Dropout(ReLU(FC2(x)))           // [N, 4096]
6.  logits <- FC3(x)                     // [N, 3]

// ResNet-50:
1.  x <- ReLU(BN(Conv7x7(x)))           // [N, 64, 96, 96]
2.  x <- MaxPool(x)                      // [N, 64, 48, 48]
3.  FOR each stage (layer1..layer4):
        FOR each bottleneck block:
            identity <- x
            x <- ReLU(BN(Conv1x1(x)))   // reduce
            x <- ReLU(BN(Conv3x3(x)))   // spatial
            x <- BN(Conv1x1(x))         // expand (no ReLU)
            IF downsample needed:
                identity <- BN(Conv1x1(identity))
            x <- ReLU(x + identity)      // residual addition
4.  x <- AdaptiveAvgPool2d(x)           // [N, 2048, 1, 1]
5.  x <- Flatten(x)                      // [N, 2048]
6.  logits <- FC(x)                      // [N, 3]

RETURN logits
```

### 4.2 Loss Function -- Weighted Cross-Entropy

```
ALGORITHM: WeightedCrossEntropyLoss
-------------------------------------------------
Input:  logits [N, K], labels [N], weights [K]
Output: loss (scalar)

1.  // Softmax probabilities (numerically stable)
    FOR i = 0 TO N-1:
        m <- max(logits[i,:])
        FOR k = 0 TO K-1:
            p[i,k] <- exp(logits[i,k] - m)
        p[i,:] <- p[i,:] / sum(p[i,:])

2.  // Weighted negative log-likelihood
    loss <- 0
    FOR i = 0 TO N-1:
        c <- labels[i]
        loss <- loss - weights[c] * log(p[i,c])

3.  loss <- loss / N

RETURN loss
```

**Mathematical Expression**:

Softmax: p(y=k|x) = exp(z_k) / sum_{j=0}^{K-1} exp(z_j)

Weighted Cross-Entropy: L = -(1/N) * sum_{i=1}^{N} w_{y_i} * log(p(y=y_i | x_i))

Where z_k is the model output (logit), w_{y_i} is the weight for class y_i.

### 4.3 Backward Pass (Backpropagation)

```
ALGORITHM: BackwardPass
-------------------------------------------------
Input:  loss (scalar), model parameters theta
Output: gradients dL/d_theta

// Chain rule with automatic differentiation:

1.  // Softmax + CrossEntropy gradient (fused)
    dL/dz_k = p_k - 1{k = y}     (for one-hot target)

2.  // FC layer gradients
    dL/dW = (dL/dz)^T * x        (weight gradient)
    dL/db = sum(dL/dz, dim=0)    (bias gradient)
    dL/dx = dL/dz * W            (input gradient)

3.  // ReLU gradient
    dL/dx = dL/dy * 1{x > 0}

4.  // BatchNorm gradient (standard PyTorch implementation)

5.  // Conv2d gradient (im2col-based -- see Section 6)
    dL/dW = dL/dY * Col(X)^T
    dL/dX = col2im(W^T * dL/dY)

6.  // ResNet residual connection gradient
    dL/dx = dL/dy + dL/dF(x)    (gradient flows through BOTH branches)
```

**ResNet Skip Connection Gradient Advantage**:

dL/dx_l = dL/dx_L * prod_{k=l}^{L-1} (1 + dF_k/dx_k)

The "1" term ensures gradient magnitude stays at least 1, greatly mitigating
the vanishing gradient problem.

### 4.4 Optimizer -- Adam

```
ALGORITHM: Adam Optimizer
-------------------------------------------------
Hyperparameters: alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, lambda=1e-4
State: m0=0, v0=0, t=0

FOR each parameter theta AT EACH STEP:

1.  t <- t + 1

2.  g <- dL/d_theta + lambda * theta        // L2 weight decay

3.  m <- beta1 * m + (1 - beta1) * g        // first moment (mean)

4.  v <- beta2 * v + (1 - beta2) * g^2      // second moment (variance)

5.  m_hat <- m / (1 - beta1^t)              // bias correction
    v_hat <- v / (1 - beta2^t)

6.  theta <- theta - alpha * m_hat / (sqrt(v_hat) + eps)
```

**Mathematical Expression**:

theta_{t+1} = theta_t - alpha / (sqrt(v_hat_t) + eps) * m_hat_t

Where:
- m_hat_t = (beta1 * m_{t-1} + (1-beta1) * g_t) / (1 - beta1^t)
- v_hat_t = (beta2 * v_{t-1} + (1-beta2) * g_t^2) / (1 - beta2^t)
- g_t = nabla L + lambda * theta_t (gradient with L2 regularization)

### 4.5 Learning Rate Scheduler -- ReduceLROnPlateau

```
ALGORITHM: ReduceLROnPlateau
-------------------------------------------------
Hyperparameters: mode="min", factor=0.5, patience=5
State: best_val_loss=inf, counter=0

After each epoch:
1.  IF val_loss < best_val_loss:
        best_val_loss <- val_loss
        counter <- 0
        best_model <- model.state_dict()
    ELSE:
        counter <- counter + 1

2.  IF counter >= patience:
        alpha <- alpha * factor      // halve learning rate
        counter <- 0
```

### 4.6 Full Training Loop

```
ALGORITHM: FullTrainingLoop
-------------------------------------------------
Input:  model, train_loader, val_loader, test_loader
        epochs = 30 (GPU) / 15 (CPU)
Output: trained_model, metrics

1.  criterion <- WeightedCrossEntropyLoss(class_weights)
2.  optimizer <- Adam(model.params, lr=1e-4, wd=1e-4)
3.  scheduler <- ReduceLROnPlateau(patience=5, factor=0.5)
4.  best_val_loss <- +inf

5.  FOR epoch = 1 TO num_epochs:

        // --- Training ---
        t_start <- time()
        model.train()
        FOR (inputs, labels) IN train_loader:
            inputs, labels <- to_device(inputs, labels)
            optimizer.zero_grad()
            outputs <- model(inputs)
            loss <- criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_time <- time() - t_start

        // --- Validation ---
        val_loss, val_acc <- evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        // --- Best model checkpoint ---
        IF val_loss < best_val_loss:
            best_val_loss <- val_loss
            save(model.state_dict(), "best_model.pth")

        history.append(train_loss, val_loss, train_acc, val_acc, train_time)

6.  // --- Final Test Evaluation ---
    model.load_state_dict(load("best_model.pth"))
    test_metrics <- full_evaluate(model, test_loader)

RETURN model, test_metrics, history
```

---

## 5. Evaluation and Metrics

### 5.1 Classification Metrics

**Confusion Matrix**:

```
                      Predicted
                 all_snow  no_snow  partial
Actual all_snow [  TP_0    FP_01    FP_02  ]
       no_snow  [  FP_10   TP_1     FP_12  ]
       partial  [  FP_20   FP_21    TP_2   ]
```

**Accuracy**: Acc = sum(TP_k) / N_test

**Precision (per-class)**: Precision_k = TP_k / (TP_k + FP_k)

**Recall (per-class)**: Recall_k = TP_k / (TP_k + FN_k)

**F1-Score (per-class)**: F1_k = 2 * Precision_k * Recall_k / (Precision_k + Recall_k)

**Macro averages**: Metric_macro = (1/K) * sum_{k=0}^{K-1} Metric_k

**AUC-ROC (One-vs-Rest)**: For each class k, treat class k as positive and all
others as negative. Compute ROC curve (TPR vs FPR at varying thresholds).
AUC = area under ROC curve (trapezoidal rule).
AUC-ROC_macro = (1/K) * sum_{k=0}^{K-1} AUC_k

```
ALGORITHM: Evaluate
-------------------------------------------------
Input:  model, test_loader, criterion
Output: metrics dict

1.  model.eval()
2.  all_labels <- [], all_preds <- [], all_probs <- []

3.  WITH no_grad():
        FOR (inputs, labels) IN test_loader:
            outputs <- model(inputs)
            loss <- criterion(outputs, labels)
            probs <- softmax(outputs, dim=1)
            preds <- argmax(outputs, dim=1)
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

4.  accuracy  <- mean(all_labels == all_preds)
    f1_macro  <- F1Score(all_labels, all_preds, average="macro")
    f1_class  <- F1Score(all_labels, all_preds, average=None)
    precision <- Precision(all_labels, all_preds, average="macro")
    recall    <- Recall(all_labels, all_preds, average="macro")
    auc_roc   <- AUC_ROC(all_labels, all_probs, multi_class="ovr")
    cm        <- ConfusionMatrix(all_labels, all_preds)

RETURN {accuracy, f1_macro, f1_class, precision, recall, auc_roc, cm, test_loss}
```

### 5.2 HPC Performance Metrics

**Throughput**: Theta = (N_samples * E) / T_total  [images/second]

**Speedup**: S(p) = T_1 / T_p = Theta_p / Theta_1

**Parallel Efficiency**: Eff(p) = S(p) / p * 100%

---

## 6. Custom CUDA Kernel Architecture

### 6.1 CUDA Programming Model

```
GPU (V100): 80 Streaming Multiprocessors (SMs)
  Each SM: multiple warps (32 threads each), 96 KB shared memory

Thread Hierarchy:
  Grid (gridDim.x, gridDim.y) --> Block (blockDim.x, blockDim.y) --> Warp (32 threads) --> Thread

Memory Hierarchy:
  Registers (per thread) --> Shared Memory (per block) --> L2 Cache --> Global Memory (HBM2)
  Access latency:  ~1 cycle        ~5 cycles           ~200 cycles    ~400 cycles
```

### 6.2 Tiled GEMM Kernel (Shared Memory)

GEMM (General Matrix Multiply) is the foundation of all convolution and FC layers.

**Operation**: C(M,N) = alpha * op(A) * op(B) + beta * C

Where op(X) = X if not transposed, X^T if transposed.

```
ALGORITHM: TiledGEMM_CUDA
-------------------------------------------------
Kernel Configuration:
  Block: (TILE x TILE) threads -- TILE in {8, 16, 32}
  Grid:  (ceil(N/TILE), ceil(M/TILE))
  Shared Memory: 2 * TILE * (TILE+1) * sizeof(float)
                 (+1 padding to avoid bank conflicts)

Each thread (tx=threadIdx.x, ty=threadIdx.y):
  row <- blockIdx.y * TILE + ty
  col <- blockIdx.x * TILE + tx

1.  acc <- 0.0

2.  FOR t = 0 TO ceil(K/TILE) - 1:

        // Load A tile into shared memory
        ak <- t * TILE + tx
        IF row < M AND ak < K:
            As[ty][tx] <- op(A)[row, ak]
        ELSE:
            As[ty][tx] <- 0.0

        // Load B tile into shared memory
        bk <- t * TILE + ty
        IF bk < K AND col < N:
            Bs[ty][tx] <- op(B)[bk, col]
        ELSE:
            Bs[ty][tx] <- 0.0

        __syncthreads()

        // Compute partial dot product
        FOR k = 0 TO TILE-1:
            acc <- acc + As[ty][k] * Bs[k][tx]

        __syncthreads()

3.  IF row < M AND col < N:
        C[row, col] <- alpha * acc + beta * C[row, col]
```

**Shared Memory Bank Conflict Prevention**:

```
Normal:  As[TILE][TILE]     --> threads may access same bank
Padded:  As[TILE][TILE + 1] --> +1 column shifts bank alignment

Example (TILE=32, 32 banks):
  Normal:  As[0][0], As[1][0], As[2][0]... all in bank 0
  Padded:  As[0][0]=bank0, As[1][0]=bank1, As[2][0]=bank2...
           Because row has 33 elements --> each row shifts by 1 bank
```

**Memory Access Analysis**:

| Method | Global Memory Accesses | Arithmetic Intensity |
|---|---|---|
| Naive GEMM | 2MNK | 0.5 FLOP/byte |
| Tiled (T=8) | 2MNK/8 | 4 FLOP/byte |
| Tiled (T=16) | 2MNK/16 | 8 FLOP/byte |
| Tiled (T=32) | 2MNK/32 | 16 FLOP/byte |

### 6.3 Im2Col Transformation

Converts convolution into GEMM by unfolding the input tensor into a column matrix.

```
ALGORITHM: Im2Col_CUDA
-------------------------------------------------
Input:  data_im [C_in, H, W]
Params: kernel (kH, kW), stride (sH, sW), padding (pH, pW)
Output: data_col [C_in*kH*kW, OH*OW]

Where: OH = (H + 2*pH - kH) / sH + 1
       OW = (W + 2*pW - kW) / sW + 1

Kernel: grid-stride loop, block_size = elem_block

FOR idx = global_thread_id TO total-1 STEP grid_stride:
    // Decompose multi-dimensional index
    ow <- idx % OW
    oh <- (idx / OW) % OH
    kw <- (idx / (OW * OH)) % kW
    kh <- (idx / (OW * OH * kW)) % kH
    c  <- idx / (OW * OH * kW * kH)

    // Input coordinates
    ih <- oh * sH - pH + kh
    iw <- ow * sW - pW + kw

    // Boundary check (zero-padding)
    IF ih in [0,H) AND iw in [0,W):
        val <- data_im[c*H*W + ih*W + iw]
    ELSE:
        val <- 0.0

    // Write to column matrix
    col_row <- c*kH*kW + kh*kW + kw
    col_col <- oh*OW + ow
    data_col[col_row * (OH*OW) + col_col] <- val
```

**Visual Explanation (3x3 kernel, stride=1, pad=0)**:

```
Input [1, 4, 4]:              Im2Col Output [9, 4]:

+---+---+---+---+             Each 3x3 patch is
| a | b | c | d |             unrolled as a column
+---+---+---+---+    ---->    in the output matrix
| e | f | g | h |
+---+---+---+---+             Column 0: [a,b,c,e,f,g,i,j,k]
| i | j | k | l |             Column 1: [b,c,d,f,g,h,j,k,l]
+---+---+---+---+             etc.
| m | n | o | p |
+---+---+---+---+
```

### 6.4 Conv2d Forward (CUDA)

```
ALGORITHM: Conv2d_Forward_CUDA
-------------------------------------------------
Input:  input [N, C_in, H, W], weight [C_out, C_in, kH, kW], bias [C_out]
Output: output [N, C_out, OH, OW]

FOR n = 0 TO N-1:
    col <- Im2Col(input[n])                           // [C_in*kH*kW, OH*OW]
    output[n] <- GEMM(weight_reshaped, col)           // [C_out, OH*OW]
      // weight_reshaped: [C_out, C_in*kH*kW]
      // GEMM_NN: no transpose on either operand
    IF bias != null:
        FOR each (c, spatial_idx):
            output[n][c][spatial_idx] += bias[c]
```

**Mathematical Expression**:

Y(n, c_out, h, w) = sum_{c_in} sum_i sum_j W(c_out, c_in, i, j) * X(n, c_in, h*s+i-p, w*s+j-p) + b(c_out)

Im2Col + GEMM formulation:

Y_(C_out, OH*OW) = W_(C_out, C_in*kH*kW) x Col(X)_(C_in*kH*kW, OH*OW)

### 6.5 Conv2d Backward (CUDA)

```
ALGORITHM: Conv2d_Backward_CUDA
-------------------------------------------------
Input:  grad_output [N, C_out, OH, OW]  (dL/dY)
        input [N, C_in, H, W]
        weight [C_out, C_in, kH, kW]
Output: grad_input, grad_weight, grad_bias

1.  grad_weight <- zeros, grad_input <- zeros

2.  FOR n = 0 TO N-1:
        col <- Im2Col(input[n])

        // Weight gradient: dL/dW += dL/dY * Col^T
        grad_weight += GEMM_NT(grad_output[n], col)  // beta=1.0 to accumulate

        // Input gradient: dL/dCol = W^T * dL/dY
        grad_col <- GEMM_TN(weight, grad_output[n])

        // Col2Im: fold gradient back to input space
        grad_input[n] <- Col2Im(grad_col)             // uses atomicAdd

3.  // Bias gradient: dL/db = sum of grad_output over (N, H, W)
    grad_bias <- sum(grad_output, dims=[0, 2, 3])

RETURN grad_input, grad_weight, grad_bias
```

### 6.6 Col2Im (Reverse Im2Col)

```
ALGORITHM: Col2Im_CUDA
-------------------------------------------------
Input:  data_col [C_in*kH*kW, OH*OW]
Output: data_im [C_in, H, W] (initialized to zero)

FOR idx = global_thread_id TO total-1 STEP grid_stride:
    ow, oh, kw, kh, c <- decompose(idx)    // same as Im2Col
    ih <- oh * sH - pH + kh
    iw <- ow * sW - pW + kw

    IF ih in [0,H) AND iw in [0,W):
        val <- data_col[col_row * (OH*OW) + col_col]
        atomicAdd(&data_im[c*H*W + ih*W + iw], val)
        // atomicAdd: multiple patches contribute to same pixel
```

### 6.7 ReLU Kernels

```
Forward:  output[i] <- max(input[i], 0)
Backward: grad_input[i] <- grad_output[i] if input[i] > 0, else 0
```

ReLU(x) = max(0, x)
dReLU/dx = 1 if x > 0, 0 otherwise

### 6.8 MaxPool2d Kernels

```
Forward:
  FOR each output position (c, oh, ow):
      max_val <- -inf, max_idx <- -1
      FOR kh, kw in kernel window:
          ih, iw <- compute input position
          IF in bounds AND input[c,ih,iw] > max_val:
              max_val <- input[c,ih,iw]
              max_idx <- linear index
      output[c,oh,ow] <- max_val
      indices[c,oh,ow] <- max_idx    // saved for backward

Backward:
  FOR each output position i:
      atomicAdd(&grad_input[indices[i]], grad_output[i])
      // gradient flows only to argmax position
```

### 6.9 Adaptive Average Pooling (1x1 output)

```
Forward:  output[c] = (1/(H*W)) * sum_{h,w} input[c,h,w]
Backward: grad_input[c,h,w] = grad_output[c] / (H*W)
```

### 6.10 Linear (FC) Layer

```
Forward:  output = input * W^T + bias                    // GEMM_NT
Backward: grad_input  = grad_output * W                  // GEMM_NN
          grad_weight = grad_output^T * input             // GEMM_TN
          grad_bias   = sum(grad_output, dim=0)
```

### 6.11 Residual Add (Skip Connection)

```
Forward:  output[i] = a[i] + b[i]
Backward: grad_a = grad_output, grad_b = grad_output
```

### 6.12 Thread Geometry Configurations (Experiment C)

| Config | GEMM Tile | GEMM Threads | Elem Block | Description |
|---|---|---|---|---|
| T32 | 8x8 | 64 | 32 | 1 warp element-wise |
| T64 | 8x8 | 64 | 64 | 2 warps element-wise |
| T128 | 8x8 | 64 | 128 | 4 warps element-wise |
| T256 | 16x16 | 256 | 256 | 8 warps GEMM + elem |
| T512 | 16x16 | 256 | 512 | 16 warps element-wise |
| T1024 | 32x32 | 1024 | 1024 | 32 warps (full block) |

GEMM Grid: grid.x = ceil(N/TILE), grid.y = ceil(M/TILE), block = (TILE, TILE)
Element-wise Grid: grid = ceil(total_elements/elem_block), block = elem_block

### 6.13 PyTorch Autograd Integration

```
ALGORITHM: PyTorch_Autograd_Integration
-------------------------------------------------

class CustomConv2dFunction(torch.autograd.Function):

    @staticmethod
    forward(ctx, input, weight, bias, stride, padding):
        output <- custom_ops.conv2d_forward(input, weight, bias,
                    stride, padding, TILE_SIZE, ELEM_BLOCK)
        ctx.save_for_backward(input, weight)
        RETURN output

    @staticmethod
    backward(ctx, grad_output):
        input, weight <- ctx.saved_tensors
        grad_input, grad_weight, grad_bias <-
            custom_ops.conv2d_backward(grad_output, input, weight,
                stride, padding, TILE_SIZE, ELEM_BLOCK)
        RETURN grad_input, grad_weight, grad_bias

// Usage:
output = CustomConv2dFunction.apply(input, weight, bias, stride, padding)
// PyTorch autograd builds the computation graph automatically
```

---

## 7. HPC Parallelization Mathematics

### 7.1 Amdahl's Law

Maximum speedup for a fixed-size problem with p processors:

S(p) = 1 / (f + (1-f)/p)

Where:
- f = serial fraction (non-parallelizable portion)
- 1-f = parallel fraction
- p = number of processors (threads)

**Estimating serial fraction from measured speedup**:

f = (1/S(p) - 1/p) / (1 - 1/p)

**Empirical results from this study**:

| Model | Measured S(28) | Estimated f | Interpretation |
|---|---|---|---|
| VGG-19 | ~10.0x | ~1.8% | Almost fully parallelizable |
| ResNet-50 | ~1.56x | ~63.4% | Large serial fraction (many small ops) |

### 7.2 Gustafson's Law

Under fixed-time assumption (problem size can grow):

S_G(p) = p - f * (p - 1)

Difference: Amdahl assumes fixed problem size (strong scaling),
Gustafson assumes fixed time (weak scaling).

### 7.3 NUMA Effects

```
Hamsi Node Architecture:
  Socket 0: Xeon 6258R, 28 cores, L3=38.5MB, ~96GB DDR4
  Socket 1: Xeon 6258R, 28 cores, L3=38.5MB, ~96GB DDR4
  Connected via UPI (Ultra Path Interconnect)

Thread Distribution:
  1-28 threads  : Socket 0 only (NUMA-local, low latency)
  29-56 threads : Both sockets (NUMA-remote access penalty ~1.5-2x)
  57-112 threads: Hyper-threading (shared physical cores)
```

### 7.4 CPU Parallelization Layers

```
ALGORITHM: CPU_Thread_Configuration
-------------------------------------------------
Input: P (desired thread count)

1.  torch.set_num_threads(P)          // ATen intra-op parallelism
    torch.set_num_interop_threads(1)  // Inter-op sequential (stable)
    OMP_NUM_THREADS <- P              // OpenMP backend
    MKL_NUM_THREADS <- P              // Intel MKL BLAS

2.  num_workers <- min(P, 8)
    // >8 workers: inter-process communication cost dominates

PyTorch CPU Parallelism Stack:
  PyTorch ATen Layer (torch.set_num_threads)
    --> OpenMP Runtime (OMP_NUM_THREADS) --> parallel for loops
    --> Intel MKL (MKL_NUM_THREADS) --> BLAS: GEMM, GEMV; LAPACK
    --> Operating System (pthreads, NUMA-aware allocation)
```

### 7.5 GPU vs CPU Speedup

S_GPU/CPU(model, threads) = Theta_GPU / Theta_CPU(threads)

| Model | CPU-1T (img/s) | CPU-28T (img/s) | GPU-V100 (img/s) | S(GPU/1T) | S(GPU/28T) |
|---|---|---|---|---|---|
| ResNet-50 | 2.35 | 3.66 | 382.7 | 162.9x | 104.6x |
| VGG-19 | 0.36 | 3.57 | 222.7 | 618.6x | 62.4x |

### 7.6 Batch Normalization

During training for each mini-batch:

mu_B = (1/m) * sum_{i=1}^m x_i
sigma_B^2 = (1/m) * sum_{i=1}^m (x_i - mu_B)^2
x_hat_i = (x_i - mu_B) / sqrt(sigma_B^2 + eps)
y_i = gamma * x_hat_i + beta

Where gamma and beta are learned parameters, eps = 1e-5.

> BatchNorm uses standard PyTorch (not custom CUDA kernels) because
> cuDNN already provides highly optimized implementations.

### 7.7 Dropout

During training: y_i = x_i / (1-p) with probability 1-p, else 0
During testing:  y_i = x_i (no dropout applied)

Where p = 0.5 (in VGG-19 FC layers). Scaling by 1/(1-p) is "inverted dropout"
-- applied during training so no change needed at test time.

### 7.8 Kaiming Weight Initialization

Custom layers use Kaiming He initialization:

W ~ Uniform(-sqrt(6/n_in), sqrt(6/n_in))

Where n_in = C_in * k_H * k_W (fan-in).

Bias: b ~ Uniform(-1/sqrt(n_in), 1/sqrt(n_in))

This initialization preserves gradient variance across layers with ReLU activations.

---

## 8. Summary Table: All Experiments

| Exp | Platform | Models | Kernels | Epochs | Batch | Measured Variable |
|---|---|---|---|---|---|---|
| A | CPU (Barbun) | VGG-19, ResNet-50 | PyTorch | 30 | 32 | Baseline timing |
| B | GPU (akya-cuda) | VGG-19, ResNet-50 | cuDNN | 30 | 32 | GPU performance |
| C | GPU (akya-cuda) | VGG-19, ResNet-50 | Custom CUDA | 30 | 16 | Thread geometry |
| D | GPU (akya-cuda) | VGG-19, ResNet-50 | cuDNN | 30 | 1-256 | Batch scaling |
| E | CPU (Hamsi) | VGG-19, ResNet-50 | MKL/OMP | 15 | 32 | Strong scaling |

---

*This document contains the architectural descriptions, pseudocode, and
mathematical background for all code in the project. Architecture diagrams
are provided as JPEG images in `paper_figures/arch_*.jpg`.*
