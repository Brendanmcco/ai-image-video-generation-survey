# ai-image-video-generation-survey
### 🔥🔥🔥 [**AI Image and Video Generation: A Survey of Models, Methods, and Challenges**](#)

> *[Brendan McCollum](#)*  
*University of Rochester*

<h5 align="center">  
 <strong>[Paper (coming soon)](#)</strong> | <strong>**[Project Page](https://github.com/Brendanmcco/ai-image-video-generation-survey.git)**</strong>
</h5>

---

## 📖 Table of Contents

- [🧠 Introduction](#-introduction)
- [🔧 Preliminaries and Foundations](#-preliminaries-and-foundations)
- [🌐 Taxonomy of Generative Models](#-taxonomy-of-generative-models)
- [📈 Key Dimensions of Progress](#-key-dimensions-of-progress)
- [🌍 Societal Impacts and Open Problems](#-societal-impacts-and-open-problems)
- [📚 References](#-references)

---

## 🧠 Introduction

This project surveys the field of **AI image and video generation**, covering foundational models, technical progress, and societal challenges. Rather than just presenting a timeline, it focuses on key methodological shifts, enabling technologies, and the growing role of these models in real-world applications.

---

## 🔧 Preliminaries and Foundations

| Concept            | Description                                                             |
|-------------------|-------------------------------------------------------------------------|
| Generative Models | Learn data distributions to synthesize new examples                     |
| Latent Space      | Compressed intermediate representation used for sampling/editing        |
| Key Techniques    | GANs, VAEs, Diffusion Models, Score-based Methods                       |
| Modality Scope    | Static image generation vs. temporally coherent video generation        |
| Datasets          | LAION-5B, WebVid-10M, ImageNet, UCF-101, MSR-VTT, etc.                  |

---

## 🌐 Taxonomy of Generative Models

### 🔹 By Output Type
- **Image Generation**
- **Video Generation**
- **Multimodal Generation** (Text-to-Image, Text-to-Video, etc.)

### 🔹 By Modeling Approach

| Type              | Examples                                                     |
|-------------------|--------------------------------------------------------------|
| Autoregressive     | PixelCNN, VideoGPT, MaskGIT                                  |
| GANs               | DCGAN, StyleGAN, StyleGAN2, StyleGAN-T                       |
| VAEs               | VQ-VAE, VQ-GAN                                                |
| Diffusion Models   | DDPM, Stable Diffusion, SVD, Imagen Video                    |
| Score-based        | Score SDE, EDM                                               |
| Hybrid/Multimodal  | DALL·E 2, Parti, unCLIP, Gen-1, Pika                         |
| 3D/Neural Fields   | GIRAFFE, DreamFusion, Sora                                   |

---

## 📈 Key Dimensions of Progress

### 🎨 Fidelity & Realism
- Metrics: FID, Inception Score, CLIPScore
- Breakthroughs: Progressive GANs, Cascaded Diffusion, Super-Resolution

### 🎛️ Controllability
- Prompt tuning, conditioning, layout control, keyframe guidance

### 🎞️ Temporal & Physical Consistency
- Consistency in object motion, scene layout, and lighting across frames

### ⚡ Efficiency & Scalability
- Fast samplers: DDIM, DiT, LCM  
- Latent diffusion for lower compute cost

### 🌍 Generalization
- Cross-domain and zero-shot generation

### 🧪 Evaluation Challenges
- Lack of standardized metrics for video
- Human preference studies vs. automated scores

### 🧰 Applications
- Creative media, data augmentation, virtual environments, simulation

---

## 🌍 Societal Impacts and Open Problems

| Topic             | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| Detection         | AI image detection tools (e.g., image forensics, watermarking)           |
| Bias & Representation | Systematic demographic imbalances in training data                  |
| Misinformation    | Deepfakes, synthetic propaganda, false visual narratives                |
| Copyright & Ownership | Unclear legal status of outputs and training data                  |
| Research Frontiers| Long videos, scene composition, real-world interaction, simulation      |

---

## 📚 References


### 1️⃣ Foundational Models and Architectures
| **Title** | **Authors** |
|-----------|-------------|
| [**Generative Adversarial Nets**](https://arxiv.org/abs/1406.2661) | Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio |
| [**Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**](https://arxiv.org/abs/1511.06434) | Alec Radford, Luke Metz, Soumith Chintala |
| [**Wasserstein GAN**](https://arxiv.org/abs/1701.07875) | Martin Arjovsky, Soumith Chintala, Léon Bottou |
| [**A Style-Based Generator Architecture for Generative Adversarial Networks**](https://arxiv.org/abs/1812.04948) | Tero Karras, Samuli Laine, Timo Aila |
| [**Analyzing and Improving the Image Quality of StyleGAN**](https://arxiv.org/abs/1912.04958) | Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila |
| [**Auto-Encoding Variational Bayes**](https://arxiv.org/abs/1312.6114) | Diederik P Kingma, Max Welling |
| [**Stochastic Backpropagation and Approximate Inference in Deep Generative Models**](https://arxiv.org/abs/1401.4082) | Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra |
| [**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239) | Jonathan Ho, Ajay Jain, Pieter Abbeel |
| [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://arxiv.org/abs/2011.13456) | Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole |
| [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer |


### 2️⃣ Text-to-Image Generation

### 2️⃣ Image-to-Image and Conditional Generation

### 4️⃣ Text-to-Video and Video Generation

### 5️⃣ Technical Innovations & Training Techniques

### 6️⃣ Applications and Tools

### 7️⃣ Evaluation and Ethics

### 8️⃣ Future Directions and Multimodal Trends

---


## 🧠 Citation

If you find this project useful, please cite our survey paper:

```bibtex
@article{your2025survey,
  title={AI Image and Video Generation: A Survey of Models, Methods, and Challenges},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
