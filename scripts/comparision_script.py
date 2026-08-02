"""
Benchmark a quantized GGUF model using llama.cpp.

Collects:
- First token latency
- Average token generation speed
- RAM usage during inference
"""
import re
import subprocess
import time
import psutil
from pathlib import Path
from helpers import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

LLAMA_CLI = ROOT_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli"
MODELS = {
    "Q4_0": ROOT_DIR / "models" / "gguf" / "Llama-3.2-3B-Q4_0.gguf",
    "Q4_HQQ": ROOT_DIR / "models" / "gguf" / "Llama-3.2-3B-Q4_HQQ.gguf",
}

LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_FILE = REPORT_DIR / "comparision_report.txt"

MAX_TOKENS = 32

PROMPTS = [
    "What is bitcoin?",
]

def extract_metrics(output: str) -> tuple[float | None, float | None]:
    prompt_match = re.search(
        r"Prompt:\s*([\d.]+)\s*t/s",
        output,
    )

    generation_match = re.search(
        r"Generation:\s*([\d.]+)\s*t/s",
        output,
    )

    prompt_speed = (
        float(prompt_match.group(1))
        if prompt_match else None
    )

    generation_speed = (
        float(generation_match.group(1))
        if generation_match else None
    )

    if prompt_speed is not None:
        logger.info(
            "Prompt Speed       : %.2f t/s",
            prompt_speed,
        )

    if generation_speed is not None:
        logger.info(
            "Generation Speed   : %.2f t/s",
            generation_speed,
        )

    return prompt_speed, generation_speed


def run_inference(process: subprocess.Popen, prompt: str, log_file: Path, model_name: str, model_path: Path) -> dict:
    """
    Run inference for a single prompt.
    """
    ps_process = psutil.Process(process.pid)

    output_chars = []
    first_token_latency = None
    peak_ram = 0
    seen_prompt = False
    start_time = time.perf_counter()
    while True:
        ch = process.stdout.read(1)

        if not ch:
            if process.poll() is not None:
                break
            continue

        output_chars.append(ch)

        try:
            ram = ps_process.memory_info().rss
            peak_ram = max(peak_ram, ram)
        except psutil.NoSuchProcess:
            pass

        # Wait until the prompt marker has appeared
        prompt_marker = f"> {prompt}"
        if not seen_prompt:
            if "".join(output_chars).endswith(prompt_marker):
                seen_prompt = True
            continue

        # First generated character after the prompt
        if first_token_latency is None and not ch.isspace():
            first_token_latency = (
                time.perf_counter() - start_time
            ) * 1000

    process.wait()

    output = "".join(output_chars)
    log_file.write_text(
        output,
        encoding="utf-8",
    )

    if process.returncode != 0:
        logger.error(output)
        raise RuntimeError("llama-cli execution failed.")

    if first_token_latency is not None:
        logger.info(
            "First Token Latency : %.2f ms",
            first_token_latency,
        )
    else:
        logger.warning(
            "Unable to determine first token latency."
        )

    logger.info(
        "Peak RAM Usage      : %.2f MB",
        peak_ram / (1024 * 1024),
    )

    prompt_speed, generation_speed = extract_metrics(output)

    logger.info("Saved log: %s", log_file.name)
    logger.info("-" * 80)

    return {
        "prompt": prompt,
        "ttft": first_token_latency,
        "prompt_speed": prompt_speed,
        "avg_token_generation_speed": generation_speed,
        "peak_ram_usage": peak_ram / (1024 * 1024),
        "model": model_name,
        "model_size_mb": model_path.stat().st_size / (1024 * 1024)
    }

def write_report(results):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as report:

        report.write("Q4_0 vs Q4_HQQ COMPARISON\n")
        report.write("=" * 80 + "\n\n")

        report.write(
            f"{'Model':<10}"
            f"{'Size(MB)':>12}"
            f"{'Prompt t/s':>15}"
            f"{'Gen t/s':>15}"
            f"{'Peak RAM(MB)':>18}\n"
        )

        report.write("-" * 80 + "\n")

        for r in results:
            report.write(
                f"{r['model']:<10}"
                f"{r['model_size_mb']:>12.2f}"
                f"{r['prompt_speed']:>15.2f}"
                f"{r['avg_token_generation_speed']:>15.2f}"
                f"{r['peak_ram_usage']:>18.2f}\n"
            )

        report.write("\n")
        report.write("=" * 80 + "\n")
        report.write("Subjective Output Quality\n")
        report.write("=" * 80 + "\n")
        report.write(
            "Compare the generated logs under the logs/ directory.\n"
        )

    logger.info("Benchmark report: %s", REPORT_FILE)

def main() -> None:
    """
    Benchmark all prompts.
    """
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    logger.info("Log Folder : %s", LOG_DIR)

    results = []
    for model_name, model_path in MODELS.items():
        logger.info("=" * 80)
        logger.info("Benchmarking %s", model_name)
        logger.info("=" * 80)
        for index, prompt in enumerate(PROMPTS, start=1):
            cmd = [
                str(LLAMA_CLI),
                "-m", str(model_path),
                "-p", prompt,
                "-n", str(MAX_TOKENS),
                "-no-cnv",
                "-st",
                "--perf",
                "--show-timings",
            ]

            logger.info("Running prompt: %s", prompt)
            logger.info("Command: %s", " ".join(cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            
            log_file = LOG_DIR / f"{model_name.lower()}_prompt_{index}.log"
            results.append(
                run_inference(
                    process,
                    prompt,
                    log_file,
                    model_name,
                    model_path,
                )
            )

    #Write report
    write_report(results)

    logger.info("Benchmark completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as ex:
        logger.exception("llama.cpp execution failed.")
        raise SystemExit(ex.returncode)
    except Exception:
        logger.exception("Unexpected error.")
        raise SystemExit(1)