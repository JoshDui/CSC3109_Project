from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
except ImportError as exc:
    raise ImportError(
        "ONNX export/evaluation requires onnx and onnxruntime. "
        "Install them with: uv add onnx onnxruntime"
    ) from exc


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model: CLIPModel, embedding_dim: int, num_classes: int):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_image_features(self, pixel_values):
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_features = self.clip_model.visual_projection(vision_outputs.pooler_output)
        return F.normalize(image_features, dim=-1)

    def forward(self, pixel_values):
        return self.classifier(self.extract_image_features(pixel_values))


class CLIPCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader: DataLoader):
        self.batches = iter(dataloader)

    def get_next(self):
        try:
            pixel_values, _, _ = next(self.batches)
        except StopIteration:
            return None
        return {"pixel_values": pixel_values.numpy().astype(np.float32)}


def collate_clip_batch(processor: CLIPProcessor):
    def _collate(batch):
        images, labels, paths = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs["pixel_values"], labels, list(paths)

    return _collate


def build_model(model_name: str, num_classes: int, checkpoint_path: Path, device: torch.device):
    clip_model = CLIPModel.from_pretrained(model_name)
    model = CLIPImageClassifier(
        clip_model=clip_model,
        embedding_dim=clip_model.config.projection_dim,
        num_classes=num_classes,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    load_clip_checkpoint(model, state_dict, checkpoint_path)
    model.eval()
    return model.to(device)


def load_clip_checkpoint(model: CLIPImageClassifier, state_dict: dict[str, torch.Tensor], checkpoint_path: Path) -> None:
    """Load either a full CLIP classifier state dict or a classifier-head-only state dict."""

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a PyTorch state dict from {checkpoint_path}, got {type(state_dict)!r}")
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as full_model_error:
        head_state_dict = normalize_classifier_head_state_dict(state_dict)
        if head_state_dict is None:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is neither a full CLIP classifier state dict nor a classifier-head-only state dict."
            ) from full_model_error
        model.classifier.load_state_dict(head_state_dict)


def normalize_classifier_head_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
    if set(state_dict) <= {"weight", "bias"}:
        return state_dict
    if all(key.startswith("classifier.") for key in state_dict):
        return {key.removeprefix("classifier."): value for key, value in state_dict.items()}
    return None


def export_fp32_onnx(model: nn.Module, output_path: Path, device: torch.device, opset: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_pixel_values = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        (dummy_pixel_values,),
        output_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def attention_matmul_nodes(onnx_path: Path):
    model = onnx.load(onnx_path)
    return [node.name for node in model.graph.node if "scaled_dot_product_attention" in node.name]


def quantize_to_int8_qdq(
    fp32_path: Path,
    int8_path: Path,
    calibration_reader: CalibrationDataReader,
    exclude_attention_matmul: bool,
    op_types_to_quantize: list[str],
):
    if exclude_attention_matmul and any(op_type.lower() == "matmul" for op_type in op_types_to_quantize):
        op_types_to_quantize = [op_type for op_type in op_types_to_quantize if op_type.lower() != "matmul"]
        print("Removed MatMul from QDQ op types. Pass --quantize-attention-matmul to quantize CLIP attention MatMul nodes.")
    nodes_to_exclude = attention_matmul_nodes(fp32_path) if exclude_attention_matmul else []
    if nodes_to_exclude:
        print(f"Leaving {len(nodes_to_exclude)} attention MatMul nodes in FP32 during QDQ quantization.")

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
    )
    onnx_model = onnx.load(int8_path)
    onnx.checker.check_model(onnx_model)


def make_session(model_path: Path, use_cuda: bool):
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif use_cuda:
        warnings.warn("--use-cuda-ort was requested, but CUDAExecutionProvider is unavailable; falling back to CPUExecutionProvider.")
    return ort.InferenceSession(str(model_path), providers=providers)


def evaluate_onnx(session: ort.InferenceSession, dataloader: DataLoader, class_names: list[str], model_label: str):
    records = []
    input_name = session.get_inputs()[0].name

    for pixel_values, labels, paths in tqdm(dataloader, desc=f"Evaluating {model_label}"):
        logits = session.run(None, {input_name: pixel_values.numpy().astype(np.float32)})[0]
        probabilities = softmax(logits)
        predictions = probabilities.argmax(axis=1)
        confidence_scores = probabilities.max(axis=1)

        for path, true_idx, pred_idx, confidence in zip(paths, labels.numpy(), predictions, confidence_scores):
            records.append(
                {
                    "path": path,
                    "true_idx": int(true_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": class_names[int(true_idx)],
                    "pred_label": class_names[int(pred_idx)],
                    "confidence": float(confidence),
                    "correct": int(true_idx) == int(pred_idx),
                }
            )

    results_df = pd.DataFrame(records)
    return results_df, summarize_predictions(results_df)


def softmax(logits: np.ndarray):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def summarize_predictions(results_df: pd.DataFrame):
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy},
            {"metric": "macro_precision", "value": macro_precision},
            {"metric": "macro_recall", "value": macro_recall},
            {"metric": "macro_f1", "value": macro_f1},
            {"metric": "weighted_precision", "value": weighted_precision},
            {"metric": "weighted_recall", "value": weighted_recall},
            {"metric": "weighted_f1", "value": weighted_f1},
        ]
    )


def save_eval_outputs(name: str, output_dir: Path, results_df: pd.DataFrame, summary_df: pd.DataFrame, class_names: list[str]):
    results_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    summary_df.to_csv(output_dir / f"{name}_summary_metrics.csv", index=False)
    report_df = pd.DataFrame(
        classification_report(
            results_df["true_label"],
            results_df["pred_label"],
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(output_dir / f"{name}_classification_report.csv")


def stratified_calibration_indices(dataset: ImageFolder, sample_count: int, seed: int = 42):
    if sample_count <= 0:
        return []
    rng = np.random.default_rng(seed)
    targets = np.asarray(dataset.targets)
    indices = []
    per_class = max(1, sample_count // len(dataset.classes))

    for class_idx in range(len(dataset.classes)):
        class_indices = np.flatnonzero(targets == class_idx)
        take = min(per_class, len(class_indices))
        indices.extend(rng.choice(class_indices, size=take, replace=False).tolist())

    remaining = min(sample_count, len(dataset)) - len(indices)
    if remaining > 0:
        selected = set(indices)
        leftover = np.asarray([idx for idx in range(len(dataset)) if idx not in selected])
        indices.extend(rng.choice(leftover, size=remaining, replace=False).tolist())

    rng.shuffle(indices)
    return indices[:sample_count]


def parse_args():
    parser = argparse.ArgumentParser(description="Export CLIP FFT model to ONNX, quantize INT8 QDQ, and evaluate accuracy.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("reports/clip_training/clip_fft_augmented/model_state.pt"),
        help="Full CLIP classifier state dict, or classifier-head-only state dict for the selected --model-name.",
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/set 12/set 12"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/val 12/val 12"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/clip_training/clip_onnx_int8_qdq"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--quant-op-types",
        default="Conv,Gemm",
        help="Comma-separated ONNX op types to quantize. Use Conv,Gemm for safer QDQ, or Conv,Gemm,MatMul for aggressive transformer quantization.",
    )
    parser.add_argument("--use-cuda-ort", action="store_true", help="Use CUDAExecutionProvider if available.")
    parser.add_argument(
        "--quantize-attention-matmul",
        action="store_true",
        help="Also quantize scaled-dot-product attention MatMul nodes. This can reduce accuracy for CLIP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", message=".*dynamic_axes.*")
    logging.getLogger().setLevel(logging.ERROR)

    project_root = Path.cwd().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    train_dataset = ImageFolderWithPaths(args.train_dir)
    val_dataset = ImageFolderWithPaths(args.val_dir)
    class_names = train_dataset.classes
    if class_names != val_dataset.classes:
        raise ValueError(f"Train and validation class folders do not match: train={class_names}, val={val_dataset.classes}")

    calibration_count = min(args.calibration_samples, len(train_dataset))
    if calibration_count <= 0:
        raise ValueError("--calibration-samples must select at least one calibration image")
    calibration_indices = stratified_calibration_indices(train_dataset, calibration_count)
    calibration_dataset = Subset(train_dataset, calibration_indices)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch(processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch(processor),
    )

    print(f"Project root: {project_root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Classes: {class_names}")
    print(f"Calibration images: {calibration_count}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"PyTorch export device: {device}")

    model = build_model(args.model_name, len(class_names), args.checkpoint, device)
    fp32_path = args.output_dir / "clip_fft_fp32.onnx"
    int8_path = args.output_dir / "clip_fft_int8_qdq.onnx"

    print(f"Exporting FP32 ONNX to {fp32_path}")
    export_fp32_onnx(model, fp32_path, device, args.opset)

    print(f"Quantizing INT8 QDQ ONNX to {int8_path}")
    calibration_reader = CLIPCalibrationDataReader(calibration_loader)
    op_types_to_quantize = [op_type.strip() for op_type in args.quant_op_types.split(",") if op_type.strip()]
    print(f"QDQ op types: {op_types_to_quantize}")
    quantize_to_int8_qdq(
        fp32_path,
        int8_path,
        calibration_reader,
        exclude_attention_matmul=not args.quantize_attention_matmul,
        op_types_to_quantize=op_types_to_quantize,
    )

    print("Evaluating FP32 ONNX")
    fp32_session = make_session(fp32_path, use_cuda=args.use_cuda_ort)
    fp32_results, fp32_summary = evaluate_onnx(fp32_session, val_loader, class_names, fp32_path.name)
    save_eval_outputs("fp32_onnx", args.output_dir, fp32_results, fp32_summary, class_names)

    print("Evaluating INT8 QDQ ONNX")
    int8_session = make_session(int8_path, use_cuda=args.use_cuda_ort)
    int8_results, int8_summary = evaluate_onnx(int8_session, val_loader, class_names, int8_path.name)
    save_eval_outputs("int8_qdq_onnx", args.output_dir, int8_results, int8_summary, class_names)

    comparison_df = pd.concat(
        [
            fp32_summary.assign(model="fp32_onnx"),
            int8_summary.assign(model="int8_qdq_onnx"),
        ],
        ignore_index=True,
    )[["model", "metric", "value"]]
    comparison_df.to_csv(args.output_dir / "onnx_accuracy_comparison.csv", index=False)
    print(comparison_df)
    print(f"Saved ONNX export and evaluation outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
