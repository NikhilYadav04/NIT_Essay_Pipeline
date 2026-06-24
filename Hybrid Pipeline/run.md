# Running the Hybrid Pipeline

This guide explains how to run single essay evaluations with the Hybrid (SRCE + MAGIC Agents) pipeline.

## Basic Execution

Run the default example (a strong essay) using the default backend (Gemini):
```bash
python hybrid_graph.py
```

## Running with Ollama (Local Model)

Make sure Ollama is running (`ollama serve`), then execute:
```bash
# Run with llama3.1:8b (recommended local model)
python hybrid_graph.py --backend ollama --model llama3.1:8b
```

## Selecting Different Essay Examples

You can evaluate different qualities of essays using the `--example` flag:
```bash
# Evaluate a weak essay
python hybrid_graph.py --backend ollama --model llama3.1:8b --example weak

# Evaluate an average essay
python hybrid_graph.py --backend ollama --model llama3.1:8b --example average

# Evaluate a strong essay
python hybrid_graph.py --backend ollama --model llama3.1:8b --example strong
```
