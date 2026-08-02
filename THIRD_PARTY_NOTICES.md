# Third-party notices

## HyperLPR3

This project includes the HyperLPR3 recognition model `models/hyperlpr3/rpv3_mdict_160_r3.onnx` and an adapter that follows the model's published preprocessing and CTC decoding route.

- Project: <https://github.com/szad670401/HyperLPR>
- License: Apache License 2.0
- Model input: BGR crop, aspect-ratio-preserving resize and zero padding to `1×3×48×160`
- Integration boundary: HyperLPR3 is used only for recognition after this project's YOLO detector selects one plate crop.

The original HyperLPR3 project and its authors retain their respective rights. Consult the upstream repository for the complete license text and notices.
