# llama-cpp-lab

Hands-on implementation of the  AI Inference using llama.cpp.

## Environment

- Windows 11
- Python 3.13
- Git
- CMake
- Visual Studio 2022 Build Tools
- Intel Core i5, with 16 GB RAM

## Clone

```bash
git clone https://github.com/rishabhom1008/llama-cpp-lab.git
cd llama-cpp-lab
```

## Model Files

Model files are **not included** in this repository due to their large size.

To reproduce the experiments:

1. Download the required model from Hugging Face.
2. Convert the model to GGUF using the provided conversion script (Task 1).
3. Quantize the GGUF model using the scripts provided for Task 1 or Task 2.
4. Place the generated GGUF files in the `models/` directory before running the examples.

## Python Setup

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

## Build llama.cpp

```bash
cd llama.cpp

cmake -B build

cmake --build build --config Release
```

---

# Task 1 - Quantize and Run a Model

## Convert Hugging Face model to GGUF

```bash
python scripts/convert_llama_gguf.py
```

## Quantize to Q4_0

```bash
python scripts/quantize_q4_0.py
```

## Run inference

```bash
python scripts/infer_benchmark.py
```

Record - reports/, logs/:

- First token latency
- Average token generation speed
- RAM usage
- Output


---

# Task 2 - Q4_HQQ Quantization

Implemented a new 4-bit quantization format (Q4_HQQ).

Modified files:

- llama.cpp\ggml\src\ggml-common.h
- llama.cpp\ggml\src\ggml-quants.h
- llama.cpp\ggml\src\ggml.c
- llama.cpp\ggml\src\ggml-quants.c
- llama.cpp\ggml\include\ggml.h

- llama.cpp\ggml\src\ggml-cpu\ggml-cpu.c
- llama.cpp\ggml\src\ggml-cpu\quants.h
- llama.cpp\ggml\src\ggml-cpu\quants.c

- llama.cpp\include\llama.h
- llama.cpp\src\llama-quant.cpp
- llama.cpp\common\arg.cpp
- llama.cpp\src\llama-model-loader.cpp
- llama.cpp\tools\quantize\quantize.cpp

## Build

```bash
cmake --build build --config Release
```

## Quantize

```bash
python scripts/quantize_q4_hqq.py
```

## Run inference and comparision between Q4_0 and Q4_HQQ

```bash
python scripts/comparision_script.py
```

COmparision report - reports/, logs/:

- Model size
- Inference speed
- Output quality

---

# Task 3 - Add a Custom CLI Flag in llama.cpp

Implemented a new CLI flag to select the backend used by the multimodal projector without affecting the base model backend.

### New CLI Flag

```bash
--mmproj-backend <backend>
```

Examples:

```bash
llama-mtmd-cli ^
  -m <model.gguf> ^
  --mmproj <mmproj.gguf> ^
  --mmproj-backend cpu ^
  -p "Describe this image"
```

```bash
llama-server ^
  -m <model.gguf> ^
  --mmproj <mmproj.gguf> ^
  --mmproj-backend cuda
```

Supported backend names depend on the backends compiled into the current llama.cpp build (for example: `cpu`, `cuda`, `vulkan`, `metal`, `hip`, `sycl`, `opencl`, etc.).

### Modified files

- llama.cpp\common\arg.cpp
- llama.cpp\common\common.h
- llama.cpp\tools\mtmd\clip.cpp
- llama.cpp\tools\mtmd\clip.h
- llama.cpp\tools\mtmd\debug\mtmd-debug.cpp
- llama.cpp\tools\mtmd\mtmd-cli.cpp
- llama.cpp\tools\mtmd\mtmd.cpp
- llama.cpp\tools\mtmd\mtmd.h
- llama.cpp\tools\server\server-context.cpp

### Build

```bash
cmake --build build --config Release
```

### Verification

Display the new CLI option:

```bash
.\build\bin\Release\llama-mtmd-cli.exe --help
```

Example:

```text
--mmproj-backend BACKEND
    Backend to use for the multimodal projector only.
    Does not affect the base model backend.
```

---

## Repository Structure

```
llama-cpp-lab/
│
├── llama.cpp/
├── models/
├── scripts/
├── reports/
├── logs/
├── requirements.txt
└── README.md
```