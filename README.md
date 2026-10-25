# Visual Similarity Search from Scratch

### Deep Metric Learning with ResNet-18, Triplet Loss, and CIFAR-100

An end-to-end image retrieval system built from scratch using PyTorch, ResNet-18, triplet-loss metric learning, custom optimizers, and exact nearest-neighbor retrieval.

The system learns 128-dimensional image embeddings and retrieves the five most visually similar images from a gallery of 6,000 CIFAR-100 test images.

**Training Status:** Completed  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB)  
**Framework:** PyTorch 2.5.1+cu121  
**Training Epochs:** 30  
**Best Checkpoint:** Epoch 30  
**Validation Triplet Accuracy:** 83.20%  
**Target Accuracy:** 75% — Achieved

---

## 1. Project Overview

This project implements an image similarity search engine without using pretrained model weights or external metric-learning libraries.

Instead of classifying an image, the network learns an embedding space where semantically related images should have smaller distances than unrelated images.

### Key Features

- ResNet-18 backbone initialized from scratch
- 128-dimensional L2-normalized embeddings
- Triplet margin loss
- Batch-hard positive and negative mining
- Random-negative mining alternative
- Custom Adam, NAdam, and Momentum SGD optimizers
- Deterministic CIFAR-100 dataset splitting
- Exact nearest-neighbor image retrieval
- Cosine similarity and Euclidean distance
- Recall@1, Recall@5, Recall@10, and mAP evaluation
- Interactive Gradio web application
- GPU-accelerated model training
- Reproducible experiment configurations

No pretrained ImageNet weights were used.

---

## 2. Dataset

Dataset: CIFAR-100

| Property | Value |
|---|---|
| Total images | 60,000 |
| Classes | 100 |
| Resolution | 32 × 32 |
| Training images | 48,000 |
| Validation images | 6,000 |
| Test images | 6,000 |

### Dataset Split

All 60,000 images from the official CIFAR-100 training and test partitions are pooled and redistributed into a deterministic 80/10/10 split.

Each class contains:

- 480 training images
- 60 validation images
- 60 test images

This split is different from the official CIFAR-100 benchmark split.

The project exports split identifiers to support reproducibility and verification.

Class labels are used to construct positive and negative relationships during triplet mining. Labels are not provided as network inputs, and no classification head or cross-entropy classification objective is used.

---

## 3. Model Architecture

The embedding network consists of:

Input Image (32 × 32 × 3)
        |
        v
ResNet-18 Backbone
(weights=None)
        |
        v
512-D Feature Vector
        |
        v
Linear Projection
(512 → 128)
        |
        v
L2 Normalization
        |
        v
128-D Unit Embedding

The baseline uses the standard ResNet-18 topology, including its original convolutional stem.

### Embedding Normalization

Given the feature representation f(x):

z = f(x) / max(||f(x)||₂, ε)

This produces embeddings with approximately unit Euclidean norm.

For normalized embeddings:

Squared Euclidean Distance = 2 - 2 × Cosine Similarity

The retrieval system ranks embeddings using squared Euclidean distance and displays Euclidean distance and cosine similarity in the interface.

---

## 4. Triplet Loss

The network is trained using triplet margin loss.

For an anchor image A, positive image P, and negative image N:

L = max(0, d(A,P) - d(A,N) + margin)

Where:

- A: Anchor image
- P: Image belonging to the same class
- N: Image belonging to a different class
- d: Euclidean distance between embeddings
- margin: Required separation between positive and negative distances

The baseline margin is 0.3.

### Batch Sampling

Training uses P × K sampling:

- P = 16 classes
- K = 4 images per class
- Batch size = 64

This guarantees that every anchor has candidate positive and negative examples within its batch.

### Mining Strategies

**Batch-Hard Mining**

For each anchor:

- Select the farthest positive embedding.
- Select the closest negative embedding.

This creates difficult triplets.

**Random-Negative Mining**

For each anchor:

- Keep the hardest-positive selection.
- Randomly select a negative from a different class.

The completed successful model uses random-negative mining.

---

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Backbone | ResNet-18 |
| Pretrained weights | None |
| Embedding dimension | 128 |
| Loss | Triplet margin loss |
| Margin | 0.3 |
| Optimizer | Custom Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epochs | 30 |
| Random seed | 42 |
| Mining | Random-negative |
| Learning-rate schedule | Cosine decay |
| Hardware | NVIDIA RTX 3050 6GB |
| Device | CUDA |

The custom Adam implementation uses first- and second-moment estimates with bias correction and coupled weight decay.

Training also includes random-crop and horizontal-flip image augmentations.

The best checkpoint is selected using validation Recall@1.

---

## 6. Experimental Results

### Successful 30-Epoch Training

The random-negative mining configuration completed all 30 epochs on the NVIDIA GPU.

### Validation Performance

| Metric | Result |
|---|---|
| Validation triplet accuracy | 83.20% |
| Required target | 75.00% |
| Target achieved | Yes |
| Best epoch | 30 |
| Final training loss | 0.19234 |

Validation triplet accuracy measures the fraction of fixed validation triplets satisfying:

d(A,P) < d(A,N)

It does not represent classification accuracy.

### Test Retrieval Performance

| Metric                 | Result |
|------------------------|--------|
| Recall@1               | 22.07% |
| Recall@5               | 47.55% |
| Recall@10              | 59.63% |
| Mean Average Precision | 11.93% |

These results were obtained using the saved best checkpoint.

Test evaluation uses all 6,000 test images as queries against the test gallery, excluding each query's own image from its ranked results.

### Training Progress

| Epoch | Validation Accuracy | Recall@1 | mAP    |
|-------|---------------------|----------|--------|
| 1     | 65.47%              | 7.87%    | 3.48%  |
| 5     | 69.45%              | 11.55%   | 5.25%  |
| 10    | 76.12%              | 15.30%   | 7.23%  |
| 15    | 79.70%              | 18.10%   | 9.13%  |
| 20    | 81.75%              | 20.83%   | 10.55% |
| 25    | 82.70%              | 21.43%   | 11.42% |
| 30    | 83.20%              | 21.60%   | 11.66% |

The table reports validation measurements recorded during training.

The final test results are reported separately because they are computed using the selected checkpoint on the held-out test set.

---

## 7. Hard Mining vs Random Mining

Two mining strategies were investigated.

### Results

| Metric                      |Hard Mining | Random Mining |
|-----------------------------|------------|---------------|
| Epochs                      |     30     |        30     |
| Initial learning rate       | 0.0001     |        0.0001 |
| Margin                      | 0.3        |        0.3    |
| Embedding dimension         | 128        |        128    |
| Validation triplet accuracy | 56.50%     |        83.20% |
| Test Recall@1               | 7.12%      |        22.07% |
| Test Recall@5               | 18.38%     |        47.55% |
| Test Recall@10              | 28.02%     |        59.63% |
| Test mAP                    | 1.94%      |        11.93% |

### Observations

Hard-negative mining exhibited unstable training behavior in the tested configuration.

The training loss approached the triplet margin, and retrieval performance deteriorated after the early epochs.

This behavior is consistent with possible embedding collapse, although embedding variance and gradient diagnostics would be needed to confirm the exact cause.

Random-negative mining produced more stable learning and progressively improved retrieval metrics.

The random-negative configuration exceeded the required 75% validation triplet accuracy target.

### Experimental Limitation

The successful checkpoint uses random-negative mining rather than batch-hard negative mining.

The original hard-mining baseline was implemented and evaluated but did not achieve the target validation accuracy.

Additional margin and embedding-dimension ablations were configured but have not been completed. No results are claimed for those experiments.

---

## 8. Evaluation Metrics

### Recall@K

Recall@K measures the fraction of queries for which at least one relevant image appears among the top K retrieved results.

Recall@K =
Queries with at least one relevant image in top K
/
Total queries

The system computes:

- Recall@1
- Recall@5
- Recall@10

### Mean Average Precision

Average Precision is calculated over the complete ranked gallery:

AP = Σ(Precision@r × Relevance(r)) / Number of relevant items

mAP = Mean(AP over all queries)

The evaluation ranks all remaining 5,999 gallery images for each test query.

---

## 9. Interactive Gradio Demo

The application provides an interactive interface for visual similarity retrieval.

### Features

1. Upload an image.
2. Generate its 128-dimensional embedding.
3. Compare the embedding against the saved image gallery.
4. Rank gallery images by similarity.
5. Display the five nearest matches.
6. Show class labels and similarity scores.

### Retrieval Pipeline

Uploaded Image
      |
      v
Image Preprocessing
      |
      v
Trained ResNet-18
      |
      v
128-D Embedding
      |
      v
Exact Distance Calculation
      |
      v
Nearest-Neighbor Ranking
      |
      v
Top-5 Similar Images

The interface displays:

- Retrieved image
- CIFAR-100 class label
- Cosine similarity
- L2 distance
- Ranking position

Uploaded images are treated as external queries. Unlike leave-one-out test evaluation, uploading an existing gallery image can produce an exact self-match.

### Demo Recording

A demonstration recording showing the working retrieval interface has been completed.

**Demo Video:** https://drive.google.com/file/d/1SRvB9uQiZLsZoOjzW5rI09LW_B65tsmC/view?usp=sharing.

For meaningful evaluation, use images with one principal object rather than multi-image collages.

---

## 10. Installation

### Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- PyTorch
- Torchvision
- NumPy
- Pillow
- Gradio
- Pytest

An NVIDIA GPU is recommended for model training.

### Windows PowerShell

Clone or download the project and open the `visual_similarity` directory.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a compatible CUDA-enabled PyTorch build if your current installation does not support CUDA.

The completed experiments used:

- PyTorch 2.5.1+cu121
- CUDA runtime 12.1
- NVIDIA GeForce RTX 3050 Laptop GPU

### Verify CUDA

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output for a working CUDA installation:

```text
True
```

### Run Tests

```powershell
python -m pytest -q
```

The project previously completed its six available tests successfully.

---

## 11. Run Training

### Random-Negative Training

The following command reproduces the configuration of the successful experiment:

```powershell
python train.py --epochs 30 --device cuda --mining random --lr 0.0001 --out runs/random_30epochs
```

### Hard-Negative Training

```powershell
python train.py --epochs 30 --device cuda --mining hard --lr 0.0001 --out runs/hard_baseline
```

This is the experimental hard-mining configuration. Previous runs showed instability, so successful convergence is not guaranteed.

### CPU Training

```powershell
python train.py --epochs 30 --device cpu --mining random --lr 0.0001 --out runs/cpu_experiment
```

CPU training is supported but substantially slower.

---

## 12. Run the Saved Model

The best trained checkpoint is available at:

```text
runs/random_30epochs/best.pt
```

Launch the Gradio application:

```powershell
python demo.py --run runs/random_30epochs
```

Open the local URL printed by Gradio.

The previously tested application launched successfully on:

```text
http://127.0.0.1:7861
```

The actual port may differ if that port is already occupied.

The model can be reused without retraining.

---

## 13. Saved Model and Outputs

The completed training run creates the following artifacts:

```text
runs/
└── random_30epochs/
    ├── best.pt
    ├── config.json
    ├── split_ids.npz
    ├── history.jsonl
    ├── gallery.npz
    └── results.json
```

### File Descriptions

| File          | Purpose                                      |
|---------------|----------------------------------------------|
| best.pt       | Best trained model checkpoint                |
| config.json   | Training configuration                       |
| split_ids.npz | Dataset split identifiers                    |
| history.jsonl | Epoch-by-epoch training history              |
| gallery.npz   | Image embeddings, labels, and gallery images |
| results.json  | Final evaluation results                     |

The checkpoint and gallery should be preserved together for the demonstration.

Dataset files, virtual environments, and generated checkpoints are not necessarily included in the source repository.

---

## 14. Reproducibility

The experiment uses:

- Fixed random seed: 42
- Deterministic dataset partitioning
- Identical dataset split definitions
- Saved configuration files
- Saved validation history
- Saved evaluation metrics
- Explicit optimizer hyperparameters

Exact numerical reproduction can vary across PyTorch versions, CUDA kernels, and hardware.

To record installed dependencies:

```powershell
python -m pip freeze > runs/environment.txt
```

The reported numbers are actual observed results, not estimated performance.

---

## 15. Limitations

The project has several limitations:

1. CIFAR-100 images contain only 32 × 32 pixels, limiting visual detail.
2. The model is trained from random initialization without pretrained features.
3. Hard-negative mining showed unstable optimization.
4. The successful model uses random-negative mining.
5. The additional margin and embedding-dimension ablations remain incomplete.
6. Exact brute-force nearest-neighbor retrieval is used instead of an approximate index.
7. Retrieval quality can decrease for images outside the CIFAR-100 distribution.
8. The reported experiments use a single training seed.

The system is a functional educational metric-learning project rather than a production-scale image search service.

---

## 16. Future Improvements

Possible future enhancements include:

- Semi-hard negative mining
- Random-to-hard mining warm-up
- Embedding-collapse diagnostics
- Additional margin ablations
- Embedding dimensions of 64 and 256
- Multiple random-seed experiments
- Faster nearest-neighbor indexing
- Larger, higher-resolution image datasets
- Better out-of-distribution query handling
- Model deployment through a hosted application

These enhancements are not claimed as implemented.

---

## 17. Conclusion

This project demonstrates an end-to-end visual similarity search pipeline using a ResNet-18 network trained from scratch.

The completed random-negative mining experiment achieved:

- **83.20% validation triplet accuracy**
- **22.07% test Recall@1**
- **47.55% test Recall@5**
- **59.63% test Recall@10**
- **11.93% test mAP**

The validation accuracy exceeded the required 75% target.

The trained embeddings were integrated into a working Gradio application capable of returning ranked visually similar CIFAR-100 images with distance and similarity scores.

The experiments also highlighted the sensitivity of deep metric learning to negative-mining strategy, with random-negative mining outperforming the tested hard-negative baseline.

The work demonstrates practical experience with computer vision, metric learning, model training, GPU acceleration, retrieval evaluation, and interactive ML application development.

---

## 18. Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18
- NumPy
- Gradio
- CUDA
- CIFAR-100
- Pytest

---

## Author

**Ayush Tiwari**

B.Tech — Data Science and Artificial Intelligence  
Indian Institute of Technology Guwahati

**GitHub:** https://github.com/Ayushdevo

**Portfolio:** https://www.ayushtiwari.tech

---

## Project Status

**Status: Trained and Demo-Validated**

- Dataset preparation: Completed
- ResNet-18 implementation: Completed
- Custom optimizer implementation: Completed
- GPU training: Completed
- Random-negative training: Completed
- Validation target: Achieved
- Test evaluation: Completed
- Gradio interface: Working
- Demo recording: Completed
- Hard-mining baseline stabilization: Incomplete
- Full ablation suite: Incomplete

All reported evaluation metrics are taken from actual training and evaluation outputs.