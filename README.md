# CSE 5524 Final

## Installation

```sh
# Create python environment
python -m venv .venv
pip install -r requirement.txt
```

## Dataset

- 3 shapes: circle (x, y, r), square (x, y, l, a), triangle (x, y, l, a)
- 8 colors: red, orange, green, purple, blue, cyan, brown, yellow
- color perturb: normal distribution of 32
- size: 64 * 64 (dimension = 4096)
- count: 1000 (~41 per category), 5000 (~208 per category), 25000 (~1042 per category)

In theory, the dataset could be compressed into [shape, colors, (r, g, b), (x, y, l, a, r)], a 10-dimension embedding for recovery.

```text
data_generation_1000.ipynb # Generation dataset of 1000
data_generation_5000.ipynb # Generation dataset of 5000
data_generation_25000.ipynb # Generation dataset of 25000
data_generation_75000.ipynb # Generation dataset of 75000
data_generation_test.ipynb # Generation test dataset of 1000
```

## Baseline

CNN-based auto encoder [auto_encoder_cnn.ipynb](./auto_encoder_cnn.ipynb)

- Add Layer Norm [auto_encoder_cnn_layernorm.ipynb](./auto_encoder_cnn_layernorm.ipynb)
- Add Residual Connection [auto_encoder_cnn_residual.ipynb](./auto_encoder_cnn_residual.ipynb)
- Add Drop Out [auto_encoder_dropout.ipynb](./auto_encoder_dropout.ipynb)
- Replace Max Pooling with Stride [auto_encoder_residual_sride.ipynb](./auto_encoder_residual_sride.ipynb)
- More Layers [auto_encoder_residual_sride_deep.ipynb](./auto_encoder_residual_sride_deep.ipynb)

Transformer-based auto encoder

- [auto_encoder_transformer.ipynb](./auto_encoder_transformer.ipynb)

Variance Auto Encoder

- CNN-based [variance_auto_encoder_cnn_residual_stride_deep.ipynb](./variance_auto_encoder_cnn_residual_stride_deep.ipynb)
- Transformer-based [variance_auto_encoder_transformer.ipynb](./variance_auto_encoder_transformer.ipynb)

## Contrastive

- This part we totally use three different kinds of models
  - CNN encoder + SimCLR
  - ViT encoder + SimCLR
  - Swin Transformer encoder + SimCLR
-  All the code in the file : [constractive learning.py](./constractive)


## Visualization

TODO

## Workload

- [x] Data generation (Yang)
- [x] Model Training
  - [x] Baseline (Yang) Targeting Apr 19
  - [x] Contrastive Learning (Adrian)
  - [x] Advanced models like ViT (Adrian)
- [x] Result Visualization (William)
- [x] Slides for presentation
- [x] Final project report & code submission
