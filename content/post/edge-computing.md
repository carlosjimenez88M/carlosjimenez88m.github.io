---
author: Carlos Daniel Jiménez
title: Edge Computing and Edge Machine Learning
date: 2024-10-14
description: A Brief Introduction to AI/Edge Computing
series:
  - edge computing
---

![](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/images/WhatsApp%20Image%202024-10-14%20at%2016.46.42.jpeg?raw=true)

### Introduction 

Data scientists often face three possibilities when deploying our products into production: through the cloud, edge computing, or the more recent variant, Edge Machine Learning. To introduce these concepts and provide context for this post, I will start by discussing the foundation—AI Computing.

### AI Computing and the Rigor of Mathematics

A promise of computing is to make machines learn, and this is the job of the data scientist. When applying a model to data, Alan Turing's insight from 1947 remains highly relevant. He simply stated that "machines learn from experience" (train set), a concept now popularized as data-driven methods.

This machine learning involves reconstructing a problem and conditions from a result to establish a mathematical formula that leads to inferences, allowing replication of the same (test set) and new results. This is where AI comes into play, defined as the mathematical process enabling operations that Machine Learning models perform, which can then be implemented through accelerated software.

But before diving into the accelerated software, which is the main topic here, it's essential to understand three key principles of AI computing to develop an efficient system:

- **ETL**: This is a classic data engineering process. Rather than explaining what it is, I want to emphasize that it needs to produce an artifact—one that provides the model with a set of knowledge.
- **Modeling**: This is where things get interesting in terms of training. It’s about understanding the model's requirements, both for training and when used in a system (whether in real-time or on a scheduled basis).
- **Inference**: Once the rules and requirements are established, we move into the process where a system is built so the model can be used and consumed for decision-making in line with its intended purpose.

To put this in simpler terms, most of our models—especially those derived from Deep Learning—must run on GPUs. These graphic cards handle a significantly higher number of mathematical operations per second compared to CPUs, leading to the term "accelerated software," which refers to the ability to create machines capable of handling a high volume of operations per second.

Now, this raises several considerations for the inference process, such as internet availability, power supply, and information capture conditions. AI computing has thus given rise to several solutions:

- **Cloud Solutions**: These are the most popular, as they eliminate the need for GPU-equipped machines or data center infrastructure to generate the necessary operations or solutions for machine learning products.
- **Edge Computing**: Processing occurs near the devices that capture the information (this ties into challenges like NLP, Computer Vision, or Audio Classification, and can be seen as part of IoT).
- **Edge Machine Learning**: Similar to edge computing, but with graphic processing capabilities to speed up the number of operations per unit of time.

In the last two points, we can talk about two types of devices: the Raspberry Pi and the Jetson Nano.

### Raspberry Pi (micro-Controler)

- Recently popularized in Edge Machine Learning, particularly for Computer Vision (though it has limited power) and software development.
- The Raspberry Pi 5 version features an ARM Cortex-A67 CPU, offering greater energy efficiency and improved performance for tasks related to ML solutions.
- It has a **Level 2 (L2) Cache of 512 KB per core** and shared **Level 3 (L3) Cache**: This improves data access speed and reduces processing times, making it highly practical for creating MLOps solutions.
- It includes an integrated VideoCore Version 7 GPU, enhancing computational performance in calculations related to Computer Vision problems.
- This device is perfect for software development and IoT solutions, though not for more complex Deep Learning tasks.

The Raspberry Pi is ideal for software development contexts, and with inference accelerators like Coral TPU, Yolo-based solutions can perform well, achieving a refresh rate of up to 15 frames per second for analysis (compared to 30 fps with the Jetson Nano).

### Jetson Nano (Edge Machine Learning)

- It features an ARM Cortex-A57 CPU, fundamental for intensive processing (the main architectural difference compared to the Raspberry Pi 5).
- Designed for parallel computing.
- It includes a **2 MB L2 Cache**: This allows for faster data access and reduces wait times during intensive task processing, which is useful in machine learning inference processes as it improves the speed of model decision-making.
- In terms of energy efficiency, it consumes about 50% more power than the Raspberry Pi 5.
- One of its most notable features is that it's designed for Deep Learning and DeepStream core solutions.

This board is a reliable companion for all types of AI solutions, including LLMs, Deep Learning, and particularly Computer Vision.

In conclusion, all these AI computing devices depend on Linux (which is essential for MLOps), and each has its own particularities. The Raspberry Pi is suited for software development, while the Jetson Nano is critical for AI programming and MLOps for intensive deployments. These two worlds are gradually converging in terms of usability and portability, while the cloud continues to innovate.

I hope you enjoyed this post—until next time!