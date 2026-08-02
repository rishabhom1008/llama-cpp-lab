import logging
import subprocess
import sys
from pathlib import Path
from helpers import get_logger

logger = get_logger(__name__)

def quantize_model(
    llama_cpp_dir: Path,
    input_gguf: Path,
    output_gguf: Path,
    quant_type: str = "Q4_0",
):
    exe = (
        llama_cpp_dir
        / "build"
        / "bin"
        / "Release"
        / "llama-quantize.exe"
    )

    if not exe.exists():
        exe = llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe"

    if not exe.exists():
        raise FileNotFoundError(f"Cannot find {exe.name}")

    if not input_gguf.exists():
        raise FileNotFoundError(f"Input model not found:\n{input_gguf}")

    output_gguf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        str(input_gguf),
        str(output_gguf),
        quant_type,
    ]

    logger.info("Running:")
    logger.info(" ".join(cmd))

    subprocess.run(cmd, check=True)

    logger.info("Quantization completed successfully.")

def main():
    root = Path(__file__).resolve().parent.parent

    llama_cpp = root / "llama.cpp"

    input_gguf = (
        root
        / "models"
        / "gguf"
        / "Llama-3.2-3B-F16.gguf"
    )

    output_gguf = (
        root
        / "models"
        / "gguf"
        / "Llama-3.2-3B-Q4_0.gguf"
    )

    quantize_model(
        llama_cpp,
        input_gguf,
        output_gguf,
        "Q4_0",
    )

if __name__ == "__main__":
    main()
