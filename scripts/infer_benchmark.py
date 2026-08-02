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
MODEL_PATH = ROOT_DIR / "models" / "gguf" / "llama-3.2-3b-q4_0.gguf"

LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_FILE = REPORT_DIR / "benchmark_report.txt"

MAX_TOKENS = 32

PROMPTS = [
    "What is bitcoin?",
    "Write a Python function to reverse a list.",
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


def run_inference(process: subprocess.Popen, prompt: str, log_file: Path) -> dict:
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
    }

def write_report(results):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open("w", encoding="utf-8") as report:

        report.write("LLAMA.CPP BENCHMARK REPORT\n")
        report.write("=" * 60 + "\n\n")

        for index, result in enumerate(results, start=1):

            report.write(f"Prompt {index}\n")
            report.write("-" * 40 + "\n")
            report.write(f"Input Prompt         : {result['prompt']}\n")
            report.write(f"First Token Latency  : {result['ttft']:.2f} ms\n")
            report.write(f"Prompt Speed         : {result['prompt_speed']:.2f} t/s\n")
            report.write(f"Generation Speed     : {result['avg_token_generation_speed']:.2f} t/s\n")
            report.write(f"Peak RAM Usage       : {result['peak_ram_usage']:.2f} MB\n\n")

        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_prompt = sum(r["prompt_speed"] for r in results) / len(results)
        avg_gen = sum(r["avg_token_generation_speed"] for r in results) / len(results)
        peak_ram = max(r["peak_ram_usage"] for r in results)

        report.write("=" * 60 + "\n")
        report.write("AVERAGES\n")
        report.write("=" * 60 + "\n")
        report.write(f"Average TTFT         : {avg_ttft:.2f} ms\n")
        report.write(f"Average Prompt Speed : {avg_prompt:.2f} t/s\n")
        report.write(f"Average Gen Speed    : {avg_gen:.2f} t/s\n")
        report.write(f"Peak RAM Usage       : {peak_ram:.2f} MB\n")

    logger.info("Benchmark report: %s", REPORT_FILE)

def main() -> None:
    """
    Benchmark all prompts.
    """
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Model      : %s", MODEL_PATH)
    logger.info("Log Folder : %s", LOG_DIR)

    results = []
    for index, prompt in enumerate(PROMPTS, start=1):
        cmd = [
            str(LLAMA_CLI),
            "-m", str(MODEL_PATH),
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
        
        log_file = LOG_DIR / f"prompt_{index}.log"
        results.append(
            run_inference(
                process,
                prompt,
                log_file,
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