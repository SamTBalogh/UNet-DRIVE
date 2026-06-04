from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that Keras .keras models can be loaded reproducibly."
    )
    parser.add_argument(
        "--models-dir",
        default="outputs/models/cv_5folds",
        help="Directory containing .keras models.",
    )
    parser.add_argument(
        "--pattern",
        default="fold_*.keras",
        help="Glob pattern used inside --models-dir.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=5,
        help="Expected number of models. Use 0 to disable this check.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help=(
            "Device policy for the check. Default is cpu to avoid consuming GPU "
            "memory in WSL2 while training uses the GPU."
        ),
    )
    parser.add_argument(
        "--cuda-malloc-async",
        action="store_true",
        help=(
            "Set TF_GPU_ALLOCATOR=cuda_malloc_async before importing Keras/TensorFlow. "
            "Only relevant with --device gpu or --device auto."
        ),
    )
    parser.add_argument(
        "--skip-registered-load",
        action="store_true",
        help="Skip load_model(path) after importing metrics.py.",
    )
    parser.add_argument(
        "--skip-compile-false",
        action="store_true",
        help="Skip load_model(path, compile=False).",
    )
    parser.add_argument(
        "--plain-clean-subprocess",
        action="store_true",
        help=(
            "Also test plain load_model(path) in a clean subprocess without importing "
            "project metrics. Failures are reported as warnings unless "
            "--require-plain-clean is used."
        ),
    )
    parser.add_argument(
        "--require-plain-clean",
        action="store_true",
        help="Make --plain-clean-subprocess failures fatal.",
    )
    parser.add_argument(
        "--_single-clean-load",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.cuda_malloc_async and args.device != "cpu":
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_tensorflow_device(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        return

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - only depends on local TF install
        print(f"WARNING: could not import TensorFlow for GPU configuration: {exc}")
        return

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"WARNING: could not enable memory growth for {gpu}: {exc}")

    if gpus:
        print(f"TensorFlow GPUs visible: {len(gpus)}")


def find_models(models_dir: Path, pattern: str, expected_count: int) -> list[Path]:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_paths = sorted(models_dir.glob(pattern))
    if not model_paths:
        raise FileNotFoundError(f"No models found in {models_dir} with pattern {pattern!r}")

    if expected_count and len(model_paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} models, found {len(model_paths)} in {models_dir}"
        )

    return model_paths


def import_project_metrics() -> None:
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import metrics  # noqa: F401


def describe_model(model) -> str:
    return (
        f"input={model.input_shape} "
        f"output={model.output_shape} "
        f"params={model.count_params()}"
    )


def clear_keras_session() -> None:
    try:
        import keras

        keras.backend.clear_session()
    finally:
        gc.collect()


def load_registered(model_path: Path) -> str:
    import_project_metrics()

    import keras

    model = keras.models.load_model(model_path)
    description = describe_model(model)
    clear_keras_session()
    return description


def load_compile_false(model_path: Path) -> str:
    import keras

    model = keras.models.load_model(model_path, compile=False)
    description = describe_model(model)
    clear_keras_session()
    return description


def single_clean_load(model_path: Path) -> int:
    import keras

    model = keras.models.load_model(model_path)
    print(describe_model(model))
    clear_keras_session()
    return 0


def run_plain_clean_subprocess(model_path: Path, args: argparse.Namespace) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_single-clean-load",
        str(model_path),
        "--device",
        args.device,
    ]
    if args.cuda_malloc_async:
        command.append("--cuda-malloc-async")

    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def check_models(args: argparse.Namespace) -> int:
    models_dir = Path(args.models_dir)
    model_paths = find_models(models_dir, args.pattern, args.expected_count)

    print(f"Models directory: {models_dir}")
    print(f"Models found: {len(model_paths)}")
    print(f"Device policy: {args.device}")
    if args.cuda_malloc_async and args.device != "cpu":
        print("TF_GPU_ALLOCATOR=cuda_malloc_async enabled")

    failures: list[str] = []
    warnings: list[str] = []

    for model_path in model_paths:
        print(f"\nChecking {model_path.name}")

        if not args.skip_registered_load:
            try:
                description = load_registered(model_path)
                print(f"  registered metrics load: OK ({description})")
            except Exception as exc:
                failures.append(f"{model_path.name}: registered metrics load failed: {exc}")
                print(f"  registered metrics load: FAILED ({exc})")

        if not args.skip_compile_false:
            try:
                description = load_compile_false(model_path)
                print(f"  compile=False load: OK ({description})")
            except Exception as exc:
                failures.append(f"{model_path.name}: compile=False load failed: {exc}")
                print(f"  compile=False load: FAILED ({exc})")

        if args.plain_clean_subprocess:
            ok, output = run_plain_clean_subprocess(model_path, args)
            if ok:
                print(f"  clean plain load: OK ({output.splitlines()[0]})")
            else:
                message = (
                    f"{model_path.name}: clean plain load failed without importing "
                    f"project metrics. This is expected for models compiled with "
                    f"the custom dice_coef metric unless compile=False is used.\n{output}"
                )
                if args.require_plain_clean:
                    failures.append(message)
                    print("  clean plain load: FAILED")
                else:
                    warnings.append(message)
                    print("  clean plain load: WARNING (failed without metrics import)")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            first_line = warning.splitlines()[0]
            print(f"- {first_line}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            first_line = failure.splitlines()[0]
            print(f"- {first_line}")
        return 1

    print("\nAll required model load checks passed.")
    return 0


def main() -> int:
    args = parse_args()
    configure_environment(args)

    if args._single_clean_load:
        return single_clean_load(Path(args._single_clean_load))

    configure_tensorflow_device(args)
    return check_models(args)


if __name__ == "__main__":
    raise SystemExit(main())
