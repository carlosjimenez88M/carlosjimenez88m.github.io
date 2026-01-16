---
title: "Raspberry Pi 16GB, Servers, and MLOps"
date: 2025-03-10
draft: false
description: "Raspberry Pi 5 (16 Gbs) like a Server"
categories: ["Edge Computing", "MLOps"]
tags: ["raspberry-pi", "mlflow", "edge-ai", "llms", "deployment"]
---


Less than two months ago, the most powerful version of the Raspberry Pi 5 hit the market, featuring 16GB of RAM. While its price ($120 USD) is a valid discussion point, as someone who uses these devices as servers for deployment testing and efficiency evaluation at the code level, I want to explore its utility from a **computer science perspective** in the context of **MLOps and LLMs testing**.

## Raspberry Pi Utility

Let's start with some common applications to build on ideas:

- **Web Server:** Particularly useful for **FastAPI** users who need a lightweight, deployable environment.
- **Deployment Testing and Task Automation:** Python users can use `cron` to schedule background execution tasks.
- **Development Server:** Access the Pi via SSH and run deployments in a **Linux environment** to monitor application status via logs.
- **AI Hat:** If equipped with an external **TPU or Coral AI**, it can be used for model training with an appropriate framework. Otherwise, its primary use is in inference rather than training.
  - The **Pi 5 features a 4-core ARM Cortex-A76 CPU at 2.4 GHz**, but it is **not optimized for ML-intensive computations**.
  - An **external GPU** can enhance its capabilities, but this requires specific configurations. **NVIDIA options**, such as **DIGITS**, can be considered.
  - **RAM remains a bottleneck** for certain deployments.

## Raspberry Pi as a Server

Since the Raspberry Pi is a **single-board microcomputer**, it serves as a **domestic server** that can be leveraged in **Edge Computing**. Regardless of the peripherals used to enhance its functionality, SSH access allows it to act as a **computational brain**—essentially, the definition of a server.

**According to Tech Craft:** “It’s the best of both worlds. Using Linux within an environment (MacOS or Windows) allows executing multiple actions that would be costly or impractical in an isolated setting.”

By using the **Pi as the computational brain**, developers can **experiment, control applications, data, and processes** running on it.

Additionally, setting up the **Pi as a NAS (Network-Attached Storage)** server allows for **file sharing via NFS**, centralizing data security, or even functioning as a **multimedia server** in areas with limited or no internet access. This is particularly useful for **home automation experiments**.

From an **application server perspective**, which is the focus of this post, **API-based servers** are of primary interest. By using the Pi for **DevOps**, it serves as a **low-scale technology testing tool**. When combined with **Docker for containerization** and **Kubernetes for orchestration**, it provides an **efficient debugging environment** for image and process testing—especially for serious **unit testing**. Additionally, **Grafana can be used** to monitor deployments.

## Raspberry Pi in MLOps

My current area of work is **Machine Learning DevOps Engineering (MLOps)**. While **DevOps** focuses on software engineering practices, **MLOps** extends this to managing the entire **ML model lifecycle**. The role of **Machine Learning DevOps Engineers** is to ensure **automation, scalability, and stability** in model deployment.

Using the Raspberry Pi for **trained model deployment** highlights the **importance of version tracking and lifecycle management**. The **focus here is inference**, especially for **LLMs that require significant RAM**.

- **With 8GB RAM**, the Pi can run **8B parameter models**.
- **With 16GB RAM**, models like **Llama 2:13B** can be deployed.

Additionally, **TensorFlow Lite** can be used for **Computer Vision, NLP, and time series models** efficiently.

From an **MLOps perspective**, automated deployments (e.g., `mlflow run .`) facilitate **model versioning and efficient release policies**. Using **Docker**, APIs and models can be **deployed, distributed, and tested**, ensuring **optimized artifacts** that prevent server overload. **Temperature control** is crucial for service reliability—especially for **high-intensity requests**.


## Raspberry Pi 5 (16GB) in LLMOps

To set up an **LLMOps environment**, follow these steps:

### 1. Install a 64-bit OS for TensorFlow/PyTorch support.
### 2. Optimize performance:

- **Cooling & Power:** The **Raspberry Pi 5** consumes more power and heats up under load (e.g., continuous inference). Use a **high-quality power supply (5V 3A min)** and **adequate cooling** (heatsink + fan or active ventilation case) to avoid *thermal throttling*.
- **CPU Governor to "performance":**

  ```bash
  sudo apt install cpufrequtils
  echo "GOVERNOR=\"performance\"" | sudo tee /etc/default/cpufrequtils
  sudo systemctl disable ondemand
  sudo reboot
  ```

- **Optimize RAM Usage:** Reduce GPU-reserved memory to 16MB using `raspi-config` (Advanced Options > Memory Split). This maximizes RAM availability for CPU and **LLM models**.
- **Fast Storage:** Use an **SSD via USB 3.0** instead of a microSD card for **faster read/write speeds**. The Pi 5 supports **M.2 NVMe storage via PCIe adapters** for **even better disk performance**.
- **Avoid Swap:** With **16GB RAM**, a **7B parameter model** should fit entirely in memory. If larger models (e.g., **13B, ~10GB RAM**) are needed, enable **zram swap** (`sudo apt install zram-tools`).

## Dependencies for LLMs

### 1. System Update:
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install essential tools:
```bash
sudo apt install -y build-essential git wget cmake python3-pip
```

### 3. Install Python dependencies:
```bash
pip install mlflow wandb llama-cpp-python fastapi uvicorn
```

### 4. Install Docker (optional for deployment):
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 5. Install Kubernetes (k3s) for orchestration (optional):
```bash
curl -sfL https://get.k3s.io | sudo sh -
```

## Running Llama 2 on Raspberry Pi

### 1. Download a **quantized** Llama 2 model (GGUF format):
```bash
mkdir -p ~/models && cd ~/models
wget -O llama2-7b-chat.Q4_K_S.gguf https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf
```

### 2. Compile **llama.cpp** (optimized for CPU inference):
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j4
```

### 3. Run an inference test:
```bash
./main -m ~/models/llama2-7b-chat.Q4_K_S.gguf -p "Hello, can you introduce yourself?" -n 50
```

## References

- [Raspberry Pi Official Website](https://www.raspberrypi.org)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Hugging Face Models](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Rockbee AI LLM on Raspberry Pi](https://rockbee.cc/pages/running-speech-recognition-and-llama-2-gpt-on-raspberry-pi)


