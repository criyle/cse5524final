# CSE 5524 Final

## Workload

- [x] Data generation (Yang)
- [x] Model Training
  - [x] Baseline (Yang) Targeting Apr 19
  - [ ] Contrastive Learning (Adrian)
  - [ ] Advanced models like ViT (Adrian)
- [ ] Result Visualization (William)
- [ ] Slides for presentation
- [ ] Final project report & code submission

## Dataset

- 3 shapes: circle (x, y, r), square (x, y, l, a), triangle (x, y, l, a)
- 8 colors: red, orange, green, purple, blue, cyan, brown, yellow
- color perturb: normal distribution of 32
- size: 64 * 64 (dimension = 4096)
- count: 1000 (~41 per category), 5000 (~208 per category), 25000 (~1042 per category)

In theory, the dataset could be compressed into [shape, colors, (r, g, b), (x, y, l, a, r)], a 10-dimension embedding for recovery.

## Baseline

CNN-based auto encoder

- Add Layer Norm
- Add Residual Connection
- Add Drop Out
- Add More Layers

Transformer-based auto encoder

Variance Auto Encoder

## Contrastive

TODO

## ViT

TODO

## Visualization

TODO
