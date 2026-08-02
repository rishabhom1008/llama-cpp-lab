from pathlib import Path
from huggingface_hub import snapshot_download
import subprocess
import sys
from helpers import get_logger

logger = get_logger(__name__)

def download_model(model_id: str, output_dir: Path):
    logger.info(f"Downloading {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

def convert_to_gguf(llama_cpp_dir: Path,
                    hf_model_dir: Path,
                    gguf_file: Path):

    script = llama_cpp_dir / "convert_hf_to_gguf.py"

    cmd = [
        sys.executable,
        str(script),
        str(hf_model_dir),
        "--outfile",
        str(gguf_file)
    ]

    subprocess.run(cmd, check=True)


def main():

    root = Path(__file__).resolve().parent.parent
    model_id = "meta-llama/Llama-3.2-3B"

    hf_dir = root / "models" / "hf" / "Llama-3.2-3B"
    gguf_file = root / "models" / "gguf" / "Llama-3.2-3B-F16.gguf"
    llama_cpp = root / "llama.cpp"

    hf_dir.parent.mkdir(parents=True, exist_ok=True)
    gguf_file.parent.mkdir(parents=True, exist_ok=True)

    download_model(model_id, hf_dir)

    convert_to_gguf(
        llama_cpp,
        hf_dir,
        gguf_file
    )

    logger.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Failed to convert model to GGUF.")
        raise

