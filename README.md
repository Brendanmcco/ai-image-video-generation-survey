# ai-image-video-generation-survey
### 🔥🔥🔥 [**AI Image and Video Generation: A Survey of Models, Methods, and Challenges**](#)

> *[Brendan McCollum](#)*  
*University of Rochester*

<h5 align="center">  
 <strong>[Paper (coming soon)](#)</strong> | <strong>[Project Page](#)</strong>
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

A curated bibliography of 50+ papers will be maintained [here](docs/references.md)  
_Add links, BibTeX entries, and categories such as "Diffusion Models", "Video Synthesis", etc._

---

## 💻 Contributing

Want to help expand the repo?  
- Add new models or papers
- Suggest examples, figures, or comparisons
- Open an issue or PR

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
