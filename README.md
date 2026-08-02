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