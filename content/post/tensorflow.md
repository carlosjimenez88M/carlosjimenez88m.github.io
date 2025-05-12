---
title: "The Decline of a Framework"
date: 2025-05-12
draft: false
description: "Reflections on TensorFlow in the context of the modern AI engine and the evolving role of Data Scientists"
categories: ["Edge Computing", "Deep Learning"]
tags: ["Tensorflow", "mlflow", "edge-ai"]
---

### Reflections on TensorFlow in the context of the modern AI engine and the evolving role of Data Scientists

Throughout my journey in the world of data, I’ve witnessed many changes — some tools fading out of popularity while others take the spotlight. R, for example, has become more niche, used mostly by statisticians and academics. Flask, once a common choice for lightweight APIs, gradually gave way to FastAPI thanks to its modularity and support for asynchronous features, redefining how APIs are designed and deployed.

While these shifts are worth noting, today’s post focuses on TensorFlow, not just as a framework, but as a case study of how the need for rapid experimentation and testing environments is reshaping the AI landscape. Ultimately, it’s a reflection on innovation and the importance of tools that are close to deployment, yet don’t require massive compute power to start building.

### The Book That Started It All
One book that significantly influenced my career as a data scientist is Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron. In fact, I own all three editions, and I frequently revisit them both for my work as an MLOps engineer—when optimizing models—and in my role as an educator.

Thanks to this book, I was able to internalize critical concepts such as:

* How do you know when a model has reached its performance ceiling?

* How do you update a model when the data distribution shifts?

This foundational knowledge led me into the world of Deep Learning, which continues to dominate the market today, especially with the rise of LLMs.

### TensorFlow: A Defining Path
That book also introduced me to TensorFlow, a robust and well-documented framework backed by Google. It proved invaluable in my early Edge Machine Learning projects—an area I remain passionate about.

But not everything was smooth. TensorFlow 1.x had several major drawbacks: static graphs made debugging difficult, the syntax lacked the clarity of idiomatic Python, and the learning curve was steep. While TensorFlow 2.x addressed many of these issues, it didn’t offer seamless migration for TF1 projects, creating additional friction within the community.

Its technical complexity and tightly coupled architecture ultimately hindered broader adoption, especially in a Python-dominated developer ecosystem.

### The Rise of PyTorch
A turning point came in May 2021, when PyTorch overtook TensorFlow in search volume. Several factors contributed to this shift:

1. NLP and LLMs: Most models on Hugging Face were built with PyTorch.

2. Academic Research: Over 95% of models on Papers with Code used PyTorch, compared to only 5% with TensorFlow.

3. Startups and Innovation: PyTorch’s dynamic graph execution, fully Pythonic design, and ease of debugging made it the go-to choice for rapid prototyping. It became the standard in agile environments, giving rise to the famous quote: "We went from experimenting in weeks to days."

### 2022: The Tipping Point
By 2022, PyTorch had clearly become the dominant framework. According to Hugging Face:

* 92% of models were trained using PyTorch.

* Only 14% supported TensorFlow.

Companies like OpenAI had already migrated completely to PyTorch by 2020, with many others following suit. Even in fields like Computer Vision, PyTorch established itself as the framework of choice (a topic for another post).

### Google’s Shift Toward JAX

In parallel, Google began focusing its efforts on JAX/Flax, a framework geared more toward research. While JAX has not yet achieved mainstream adoption or stood out in model competitions, its emergence marked a strategic pivot—Google began moving away from TensorFlow as its default deep learning platform.

A telling sign: the official TensorFlow YouTube channel significantly reduced its publishing frequency, and newer Google research releases were already being built with JAX.

### Keras’ New Role: Beyond TensorFlow
One final point worth noting is the evolution of Keras. With version 3, it’s no longer exclusive to TensorFlow. It is now multi-backend, supporting PyTorch, JAX, and of course, TensorFlow. This change allows tools like Keras Tuner to be used across different frameworks, promoting more flexible, framework-agnostic development workflows.

### Why TensorFlow Lost Ground (My Perspective)
Taken together, these shifts led to TensorFlow’s gradual decline in both research and production. Despite Google’s powerful TPU infrastructure—which works well with TensorFlow—the broader market and research community moved in a different direction.

To me, the breaking point wasn’t just the lack of backward compatibility or internal complexity—it was the lack of Pythonic elegance. In a field dominated by Python developers, this became a critical flaw.

The battle may have been lost—not because TensorFlow lacked potential, but because the needs of the ecosystem evolved faster than the framework itself.


### A fun fact (interesting tidbit)
There's something that caught my attention  I'm going to present two graphs. The first one shows the trend according to Google Trends, where the blue line is TensorFlow and the red one is PyTorch. I had talked about the decline in usage of the framework, and you can see how it's falling, but what particularly catches my attention is that there's a downward trend for PyTorch in 2025. I don't know if it has something to do with Rust, or if researchers and developers have fewer publications, considering we're in an era where model releases or versions are an everyday occurrence.
![](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/img/comparative.png?raw=true)

In the second graph, also obtained from Google Trends, something else caught my attention even more: In Latin America, Africa, and Eastern Europe, TensorFlow still predominates (there's something to infer from that), although in India's case, usage is 59% TF and the rest PyTorch, which means that in industry it continues to be applied aggressively. It would be worth analyzing which industries these are. I don't know if it has something to do with Vision Language Models, but well, these are the data points and they should be shared.
![](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/img/tf2.png?raw=true)


### Final Thoughts
* Deep Learning reached a stage where graph flexibility and computational efficiency became decisive factors in framework adoption.

* As in all things, technological Darwinism prevailed—driven by LLMs, agentic systems, and RAG-based architectures, experimentation speed and model development agility became more valuable than raw power.

* Startups and the industry as a whole paved the path toward innovation, and while TensorFlow fought many worthy battles, its future might lie in niche areas like domotics and Edge AI.

* The key takeaway is this: in machine learning, development speed and team-wide standardization outweigh personal preferences. I may personally enjoy working with TensorFlow, but in production environments, that choice could come at a cost if it’s not aligned with the team's common stack.

That’s all for this entry—
I hope you enjoyed these reflections!