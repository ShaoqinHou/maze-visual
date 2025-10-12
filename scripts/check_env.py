import importlib
import sys


def main():
    missing = []
    versions = {}

    pkgs = [
        ("torch", "__version__"),
        ("torch_geometric", "__version__"),
        ("numpy", "__version__"),
        ("tqdm", "__version__"),
        ("yaml", "__version__"),  # PyYAML
        ("networkx", "__version__"),
        ("tensorboard", "__version__"),
    ]

    print("Package versions:")
    for name, attr in pkgs:
        try:
            m = importlib.import_module(name)
            versions[name] = getattr(m, attr, "?")
        except Exception as e:
            versions[name] = f"NOT INSTALLED: {e.__class__.__name__}: {e}"
            missing.append(name)
        print(f"- {name}: {versions[name]}")

    # Torch / CUDA checks
    try:
        import torch
        print("\nTorch/CUDA:")
        print(f"- torch.version.cuda: {torch.version.cuda}")
        print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"- torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
                print(f"- torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
            except Exception as e:
                print(f"- GPU query error: {e}")
    except Exception as e:
        print(f"Torch check failed: {e}")

    # PyG ops check
    try:
        from torch_geometric.utils import group_argsort, scatter, softmax
        print("\nPyG ops import: OK (group_argsort, scatter, softmax)")
    except Exception as e:
        print(f"PyG ops import failed: {e.__class__.__name__}: {e}")

    print("\nSummary:")
    if missing:
        print("Missing packages:", ", ".join(missing))
        sys.exit(1)
    else:
        print("All listed packages are installed.")


if __name__ == "__main__":
    main()

