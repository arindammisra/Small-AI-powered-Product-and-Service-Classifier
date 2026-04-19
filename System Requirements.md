System Requirements: AI-Powered UNSPSC Classifier
This document outlines the hardware, software, and dependency requirements necessary to successfully deploy and run the local, GPU-accelerated UNSPSC classification system. The system relies on purely LLM-based hierarchical tree traversal and lexical BM25 indexing, requiring specific compute resources to ensure low-latency inference.

1. Hardware Requirements
To support local execution of large language models alongside image processing capabilities, a dedicated GPU is strictly required.

GPU (Graphics Processing Unit): NVIDIA GPU with a minimum of 8GB to 12GB VRAM (e.g., RTX 3060 12GB, RTX 4070, RTX 3080, or enterprise equivalent). 12GB+ is highly recommended to comfortably run the default gemma3:12b or multimodal qwen2.5vl:7b models without heavy quantization bottlenecks or Out-of-Memory (OOM) errors.

System RAM: 16GB minimum (32GB recommended). Building the hierarchical trees and computing the BM25 Okapi indices in memory from the full UNSPSC dataset requires sufficient overhead alongside OS processes.

Storage: SSD (Solid State Drive) is required. You will need approximately 15–20GB of free space for caching the local LLM weights via Ollama, plus space for the UNSPSC .xlsx taxonomy file.

CPU: Modern multi-core processor (Intel Core i5/AMD Ryzen 5 or better) to handle the Tkinter mainloop, BM25 text preprocessing, and image thumbnail generation concurrently.

2. Software & Operating System
Operating System: Windows 10/11 (natively matching the C:\Users\... hardcoded path architecture), though the code is inherently cross-platform and adaptable to Linux or macOS with minor path modifications.

Python Environment: Python 3.8 or higher.

LLM Backend Server: Ollama must be installed and running as a background service on the local machine (http://localhost:11434).

3. Model Requirements
The models designated in the script must be pulled locally via Ollama prior to execution. Depending on your active variable selection, run the appropriate command(s) in your terminal:

Primary Text/Classification Model: ollama run gemma3:12b (or gemma3:4b for lower-VRAM environments).

Vision/Image Processing Model: ollama run qwen2.5vl:7b (required if USE_IMAGE_MATCHING = True and image inputs are utilized).

4. Python Package Dependencies
The following packages must be installed in your Python environment. A requirements.txt file should include:

pandas (For reading and processing the taxonomy data structure)

openpyxl (Required backend for pandas to parse .xlsx files)

requests (For communicating with the local Ollama API endpoint)

rank_bm25 (For executing the lexical fallback search against the commodity metadata)

Pillow (For image compression, resizing, and rendering in the UI)

tk (Usually bundled with standard Python installations; handles the graphical interface)

5. File System Setup
Taxonomy Database: The official UNSPSC taxonomy dataset must be downloaded and stored at the exact path referenced in the script: C:\Users\arind\Downloads\Classifier UNSPSCv2\unspscfull.xlsx. The Excel file must contain the specific columns: Segment, Segment Title, Segment Definition, Family, Family Title, Family Definition, Class, Class Title, Class Definition, Commodity, Commodity Title, and Commodity Definition.