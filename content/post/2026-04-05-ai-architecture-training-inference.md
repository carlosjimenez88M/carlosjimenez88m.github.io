---
author: Carlos Daniel Jiménez
date: 2026-04-05
title: "AI Architecture - Notions on Training and Inference"
draft: false
description: "A technical breakdown of CPU, GPU, TPU, and Edge AI hardware tradeoffs for training and inference workloads — with real-world cost data and a deep dive into Raspberry Pi 5 + Hailo-10H."
categories: ["engineering"]
tags: ["cpu", "gpu", "tpu", "edge-ai", "inference", "deep-learning", "nvidia", "hardware", "raspberry-pi", "mlops", "hailo", "edge-computing"]
---

CPU · GPU · TPU · Edge Computing

*The problem is not that AI is expensive. It's that for years, people paid to train models as if that were the main cost — when the real cost, the one that never stops, is serving every response.*

## TL;DR

- Inference costs exceed training by **15x–20x** over a model's operational lifetime. Optimizing for training while ignoring inference is optimizing the wrong problem.
- **CPU (Intel Xeon AMX)**: the correct choice when the model lives alongside the data. Network latency kills any compute gain from moving to a GPU cluster.
- **NVIDIA GPU (Blackwell/Hopper + TensorRT-LLM)**: still the default for research and heterogeneous production. CUDA is a 20-year moat. Don't lock in at peak prices.
- **Google TPU v6/v7**: the right answer for high-volume, predictable inference. Midjourney cut monthly costs from $2.1M to $700K. The CUDA migration barrier no longer exists.
- **Edge AI**: thermodynamics, not algorithms, sets the limits. Pi 5 + Hailo-10H delivers 320 ms TTFT (6.4× faster than CPU-only) with a PCIe x1 bottleneck you need to design around.
- The right hardware is not the most powerful. It is the one that matches the problem topology to the silicon architecture without wasting energy or budget.

---

## Introduction

In 2023, Nvidia published a post titled **What Is AI Computing?** focused on handling intensive computations — particularly useful for embedding design and optimization processes in Machine Learning — and advancing toward hardware acceleration to find patterns in immense amounts of data, thereby updating the assumptions of ML or AI models. All of this typically runs on GPUs.

Let's hold onto the GPU as the vehicle for compute acceleration. In the context of Deep Learning, this matters especially when model architectures are complex (and difficult to move if you're working with TF), primarily because they are grounded in linear algebra equations, which dictate how information is hierarchically processed. Entropy then takes on special significance for those studying variance behavior over models — genuinely interesting from a scientific standpoint. The point is this: with greater volumes of data and computation, variance that was once distorted can eventually be correctly interpreted when normalized over a new wave of data from the same source. That is what makes data drift a discipline worth mastering.

This sets the stage for discussing accelerated software development — faster mathematical processes — through specialized hardware: TPUs, CPUs (in certain cases, which I'll expand on), and the classic GPU.

## I. The Problem Nobody Wanted to See

The industry built temples to training. GPU clusters, scaling papers, benchmark records. But training is an event: it happens, it ends, and the capital invested becomes fixed. Inference, on the other hand, **never ends — it can be used indefinitely**. It scales with every user, every token, every prompt sent at three in the morning.

The 2026 data is unambiguous: over the operational lifetime of a model in production, inference costs exceed training by a factor of **15x to 20x**. A billion dollars spent on training becomes fifteen to twenty billion spent on serving. This is not an academic curiosity. It is the equation that determines whether an AI project is financially viable.

| | **15–20×** inference vs. training, in cumulative operational costs · *The 'hidden iceberg': training is only the visible tip* | |
| --- | --- | --- |

This is why data center design has migrated from a compute discipline to one of *thermodynamics and finance*. The choice between CPU, GPU, TPU, or edge accelerator is not primarily a technical decision — it is a budgetary one. A mismatched architecture is not just slow — it is ruinous.

## II. The CPU: Modest but Indispensable

Saying that CPUs "returned" to AI is inaccurate. They never left. What changed is that the argument for using them in inference stopped being pragmatic and became mathematically sound.

### Advanced Matrix Extensions (AMX): The Real Shift

Sixth-generation Intel Xeon processors — Granite Rapids — integrate **Advanced Matrix Extensions (AMX)** directly into each core. Unlike AVX-512, which operates on one-dimensional vectors, AMX implements a two-dimensional matrix multiplication engine embedded within the core itself. The practical result: MLPerf Inference v6.0 shows that a Xeon 6 can serve low-latency inference on Llama 3 8B in real time.

The historical bottleneck was not compute but memory (a topic I'll cover in a separate post). Granite Rapids systems with MRDIMM modules at 8,800 MT/s and UPI 2.0 interconnects at 24 GT/s have largely resolved that problem. PyTorch 2.6 adds native Float16 support for x86, halving memory consumption during inference and accelerating execution via **torch.compile** + the Inductor backend — without rewriting any code.

### When a CPU Makes Sense

| | **Rule of thumb** — *If the model lives alongside the data — database, ERP, record system — moving it to a GPU cluster introduces network latency that destroys any compute gain. Processing on the same socket wins.* |
| --- | --- |

For sporadic workloads, the GPU sits idle 80% of the time while the billing clock runs. A Xeon 6 does not have that problem. The TCO argument is straightforward: if you cannot saturate the GPU, there is no justification for renting it.

## III. The GPU: The Software Moat Nobody Has Crossed

NVIDIA's real power is not in its transistors. It is in twenty years of CUDA — an ecosystem of compilers, libraries, and documentation that no competitor has replicated in depth. Switching GPU hardware means switching hardware. Escaping CUDA means switching engineering culture.

### Blackwell: The Numbers and What They Mean

The B200 chip operates with 192 GB of HBM3e, 8.0 TB/s of memory bandwidth, and reaches 4,500 TFLOPS in dense FP8 — double the H200. The full GB200 NVL72 rack clusters 72 accelerators that function as a single entity. The entry price: power densities so extreme that **direct liquid cooling** is a minimum requirement, not an option. Data centers built for air cooling are becoming physically obsolete.

**Data Center GPUs — Current State (2026)**

| **Chip** | **VRAM** | **Bandwidth** | **TFLOPS FP8** | **Differentiator** |
| --- | --- | --- | --- | --- |
| **NVIDIA B200 (Blackwell)** | 192 GB HBM3e | 8.0 TB/s | 4,500 | Maximum flexibility, CUDA |
| **NVIDIA H200 (Hopper)** | 141 GB HBM3e | 4.8 TB/s | ~1,979 | Established clusters |
| **NVIDIA H100 (reference)** | 80 GB HBM3 | 3.35 TB/s | ~990 | Active secondary market |

### The Problem Benchmarks Don't Show

GPUs were designed for training: dynamic topologies, changing computational graphs, total flexibility. In large-scale, deterministic inference, that flexibility becomes overhead. **Dynamic branching, redundant host-device copies over PCIe, energy efficiency measured in 'tokens per watt'** — here the GPU loses ground to specialized hardware.

| | **Economic volatility — a real case** · *H100 rental prices fell between 64% and 75% over fourteen months: from $8–10/hour to $2.99/hour. Companies that signed long contracts at the peak paid more than triple the market rate. Committing to a GPU provider at 2023 prices was, in many cases, a financially catastrophic decision.* |
| --- | --- |

### TensorRT-LLM and PagedAttention: The Real Differentiators

TensorRT fuses network layers, applies kernel autotuning, and calibrates low-precision formats (INT8, FP4) with minimal accuracy degradation. Triton Inference Server separates the prefill and decode phases — an architectural optimization that fundamentally changes the latency equation under high load. PagedAttention eliminates KV-cache fragmentation. These are not implementation details: they are why the GPU remains the default choice for research and heterogeneous production environments.

## IV. TPU: When Massive Inference Changes the Arithmetic

Google built TPUs for its own systems — search, recommendation, translation — because in that context inference never stops. The design emerged from an operational necessity, not an academic exercise. That difference in origin explains why TPUs are structurally distinct from GPUs, and why that distinction matters.

### The Systolic Array: An Honest Architecture

TPUs replaced generic control units with **systolic arrays**: grids of arithmetic-logic units where data flows synchronously, computing dot products without accessing memory on every operation. The result is energy efficiency per operation that general-purpose GPUs cannot match under continuous inference loads.

The Trillium generation (TPU v6) adds **SparseCore v3** — accelerators dedicated exclusively to embedding tables, which dominate RAG systems and recommendation workloads. A 2x to 3x advantage over generic hardware on those specific tasks. The TPU v7 (Ironwood) scales up to pods of **9,216 chips** interconnected via fiber optics with dynamic circuit switching, rerouting traffic in milliseconds under node failures.

### Midjourney: The Case That Settled the Debate

TCO arguments are easy to construct with convenient assumptions. Midjourney's case needs none.

| | **65%** monthly operational savings — TPU v6e migration (Midjourney, 2025/2026) · *From $2.1M to $700K/month · Payback: 11 days · User downtime: 0* | |
| --- | --- | --- |

Midjourney was serving image generation (Stable Diffusion XL, Flux models) on A100 and H100 clusters with monthly bills exceeding $2.1 million. The migration to TPU v6e reduced that cost to under $700,000 — a sustained annual saving of $16.8 million. Price-performance improved **4.7x** and energy consumption dropped 67%. The cost of rewriting the code was recovered in **11 days** of operation. Zero user downtime.

What this case demonstrates is not that TPUs always win. It is that GPU vendor lock-in can be expensive, and that the alternative is technically and commercially viable.

### JAX and PyTorch/XLA: The Migration Barrier No Longer Exists

The historical hesitation around TPUs was legacy CUDA code. In 2026, that wall has come down. **JAX over XLA** operates with 'whole-program analysis': the compiler inspects the entire network before execution, fuses hundreds of operations into super-operators, and eliminates redundant data transfers. With primitives like *pmap* or *shard_map*, code written for one chip scales to thousands without re-engineering.

For those who prefer not to abandon PyTorch: **PyTorch/XLA 2.6** enables TPU orchestration with minimal changes. The new *scan* operator (inspired by *jax.lax.scan*) eliminates disproportionate compile times caused by decode loops in large models. Torchax and vLLM complete the bridge. The "we can't migrate because of the code" argument is, in most cases, no longer true.

## V. Edge AI: The Limit Is Physics, Not the Algorithm

While the industry builds data centers the size of small cities, there are applications where depending on the cloud is a design error. An autonomous vehicle that needs network latency to brake is not an AI system — it is a betting system. Surgical robotics, industrial inspection in environments without connectivity, perimeter surveillance that cannot exfiltrate data — all of these demand local compute.

The field has matured and moved away from trying to compress hundred-billion-parameter LLMs into miniature devices. The current consensus orbits around **specialized SLMs and VLMs**, designed to operate within thermal budgets ranging from microwatts to a few watts.

### The Wall That Isn't Software

Edge limits are not imposed by algorithmic incapacity. They are imposed by thermodynamics. An inspection drone runs on battery: every extra watt of compute reduces mission time. A fanless chassis accumulates heat under sustained load and enters thermal throttling — FPS drops, latency SLAs are violated. There is no software patch for the second law of thermodynamics.

The industry's technical responses deserve attention: **DVFS (Dynamic Voltage and Frequency Scaling)** coupled with **early exit networks** — the model stops compute at shallow layers if confidence exceeds a threshold, immediately powering down the accelerator. **Sony's IMX500 architecture** goes further: it computes directly on the sensor's focal plane, eliminating raw frame transfers between camera and processor. It achieves the highest possible computational density by removing the slowest bus in the system.

### The Accelerator Ecosystem: Surgical Selection

**Edge Accelerators — Practical Comparison (2026)**

| **Platform** | **Performance** | **Typical Power** | **Honest Use Case** |
| --- | --- | --- | --- |
| **NVIDIA Jetson AGX Orin** | 275 TOPS | 10–60 W | Industrial robotics, sensor fusion, when CUDA is a hard requirement |
| **Axelera Metis M.2 Max** | 214 TOPS | 20–40 W | High-resolution multi-camera, VLMs, visual IoT at scale |
| **NVIDIA Jetson Orin Nano** | 40–67 TOPS | 7–15 W | Drones, autonomous cameras, CUDA heritage in small form factor |
| **SiMa.ai MLSoC** | 50+ TOPS | < 5 W | Industrial edge where every milliampere counts |
| **Hailo-10H** | 40 TOPS (INT4) | < 5 W | Local SLMs, LLM prefill, private generative AI |
| **Hailo-8 / 8L** | 13–26 TOPS | 2.5–3 W | Always-on vision, low-cost smart cameras |

The Jetson AGX Orin is the right processor for applications where CUDA heritage is non-negotiable and the energy budget allows 60W. **SiMa.ai** won the MLPerf ResNet-50 Single Stream benchmark against larger competitors while operating below 5W — not through magic, but through a memory architecture that minimizes the energy cost of every internal transfer. **Hailo-10H** targets specifically the LLM prefill phase at the edge, where compute is dense and predictable — the ASIC's strong suit.

## VI. Raspberry Pi 5 + Hailo-10H: Real Power with a Real Bottleneck

The Raspberry Pi 5 did something architecturally important that its predecessors could not: it exposed a native PCIe bus to the outside world. That changed its category. It is not just an educational board — it is an embedded platform with a direct interconnect to specialized hardware, without the USB bridge bottleneck.

In Q1 2026, Raspberry Pi launched the **AI HAT+ 2**: an expansion board that couples the Hailo-10H chip to the board, delivering 40 TOPS in INT4 format and — crucially — **8 GB of LPDDR4X memory soldered directly onto the accelerator's substrate**. The dedicated memory solves the problem that sank previous generations: the Hailo-8 competed with the CPU for the Pi's scarce RAM bus. With its own memory, the accelerator operates without contention.

### The Bottleneck That 40 TOPS Cannot Hide

This is where technical honesty matters. The Raspberry Pi 5 has two PCIe paths:

**Bus 0002 (x4):** four lanes to the RP1 south bridge, reserved for USB 3.0, Gigabit Ethernet, and GPIO. Not available for external peripherals.

**Bus 0001 (x1):** the only path available for the external FFC connector — and therefore for the HAT. Hardware-limited to PCIe Gen 3.0 x1.

| | **~1 GB/s** real bandwidth available to the Hailo-10H on Raspberry Pi 5 · *PCIe Gen 3.0 x1 · 8.0 GT/s theoretical · symmetric direction* | |
| --- | --- | --- |

The Hailo-10H can execute 40 tera-operations per second internally. The channel connecting it to the CPU transfers, at best, 1 GB/s. **The gap between the chip's internal throughput and its connection to the outside world is the system's real limit.** Autoregressive inference — where each generated token requires a new round trip over that channel — collapses quickly. The bottleneck is the bus, not the accelerator.

| | **Technical diagnosis** — *The PCIe x1 bottleneck can be confirmed by inspecting dmesg on Linux. The software fix is to explicitly force Gen 3.0 in config.txt with `dtparam=pciex1_gen=3`, and install the `hailo-all` package for firmware, kernel modules, and the HailoRT runtime.* |
| --- | --- |

### Where It Works — and Where It Doesn't

The Pi 5 + Hailo-10H combination works well for **bulk prefill** — processing input context, text or audio embeddings, visual detection. These are dense, predictable operations that the chip can absorb internally. The result: *Time to First Token* (TTFT) for QWEN2.5-1.5B in INT4 format drops from **2,039 ms** (CPU-only with llama.cpp) to **320 ms** with offload to the Hailo-10H. A 6.4x improvement in the startup latency perceived by the user.

Where it **does not** work well: sustained autoregressive generation, where each token requires accessing the KV-cache and sending activations over the x1 bus. The 1 GB/s channel becomes a permanent bottleneck. **The correct architecture for Pi + Hailo is: Hailo handles prefill, the CPU handles token-by-token decoding**, and the PCIe channel transfers only compressed context, not raw activations.

**TTFT — CPU only vs. Hailo-10H offload (QWEN2.5-1.5B, 96 context tokens)**

| **Configuration** | **TTFT (ms)** | **Difference** |
| --- | --- | --- |
| **llama.cpp on CPU (Raspberry Pi 5 only)** | 2,039 ms | Baseline |
| **Prefill on Hailo-10H (dedicated LPDDR4X memory)** | 320 ms | −84% · 6.4× faster |

### Industrial Use Cases That Work Today

**Husqvarna Automower**: autonomous outdoor navigation where GPS fails and Wi-Fi doesn't exist. The Hailo processes cameras and sensors locally, in real time, with no network dependency.

**B&R industrial cameras**: visual shrinkage analysis and irregularity identification in closed retail environments. Local processing eliminates the need to stream H.264 footage to remote data centers — along with the privacy implications that entails.

### What Raspberry Pi 6 Would Need to Solve

The technical community has two non-negotiable demands for the next cycle: native system memory reaching 32 GB, and an M.2-enabled PCIe Gen 4.0 x4 slot. Those two changes would eradicate the 1 GB/s bottleneck and turn the platform into a compact inference server with mid-scale capabilities. Without them, the HAT+ 2 will remain a powerful tool with a clear structural ceiling.

## VII. Conclusions Without Euphemisms

There are four questions that determine hardware selection, and only one of them is technical:

**1. Does the model live alongside the data?** CPU with AMX. The cost of moving data across the network exceeds any advantage from specialized compute.

**2. Is maximum flexibility needed for active research or heterogeneous production?** NVIDIA GPU, with TensorRT-LLM and vLLM. The CUDA ecosystem is a de facto monopoly, and in many contexts it is the right resource. Just don't commit at peak prices.

**3. Is the product large-scale, predictable, high-volume inference?** TPU. The Midjourney case needs no additional commentary. The CUDA migration wall no longer exists.

**4. Must compute happen on-device, without cloud, under energy constraints?** Edge AI with the appropriate accelerator. Pi 5 + Hailo-10H for low-cost projects with microwatt energy budgets. Jetson AGX Orin when CUDA is a hard requirement and the energy budget allows 60W.

What the industry learned in 2026 — and should have learned earlier — is that the right hardware is not the most powerful. It is the one that matches the problem topology to the silicon architecture without wasting energy or budget on capabilities that will never be used. Ubiquitous AI will not come from larger data centers. It will come from more honest hardware.

## Quick Reference: Hardware Decision Tree

**Selection Guide by Workload Profile**

| **If your case is…** | **Primary choice** | **Main reason** |
| --- | --- | --- |
| Inference next to a database or ERP | CPU + AMX (Xeon 6) | Avoids network latency, lower TCO without idle GPU |
| Research, changing models, legacy CUDA | GPU NVIDIA (Blackwell/Hopper) | Maximum flexibility, mature TensorRT ecosystem |
| Massive inference, high volume, predictable batch | Google TPU v6/v7 | 4.7× price-performance vs GPU, 67% less energy |
| Edge vision, heavy robotics, CUDA required | NVIDIA Jetson AGX Orin | 275 TOPS, complete JetPack ecosystem |
| Ultra-efficient edge, <5W, industrial | SiMa.ai MLSoC | Best TOPS/W ratio on the market |
| Local SLM, privacy, fast prefill, low cost | Raspberry Pi 5 + Hailo-10H | 320 ms TTFT, 8 GB dedicated memory, $0 cloud |
