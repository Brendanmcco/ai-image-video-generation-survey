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
| **Title** | **Authors** |
|-----------|-------------|
| [**Zero-Shot Text-to-Image Generation**](https://arxiv.org/abs/2102.12092) | Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever |
| [**Hierarchical Text-Conditional Image Generation with CLIP Latents**](https://openai.com/index/dall-e-2/) | Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen |
| [**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**](https://arxiv.org/abs/2112.10741) | Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen |
| [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer |
| [**CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers**](https://arxiv.org/abs/2204.14217) | Ming Ding, Wendi Zheng, Wenyi Hong, Jie Tang |
| [**AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks**](https://arxiv.org/abs/1711.10485) | Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He |
| [**Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors**](https://arxiv.org/abs/2203.13131) | Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, Yaniv Taigman |
| [**DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models**](https://arxiv.org/abs/2210.08933) | Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, Lingpeng Kong |
| [**Video Diffusion Models**](https://arxiv.org/abs/2204.03458) | Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet |

### 2️⃣ Image-to-Image and Conditional Generation
| **Title** | **Authors** |
|-----------|-------------|
| [**Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)**](https://arxiv.org/abs/1611.07004) | Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros |
| [**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)**](https://arxiv.org/abs/1703.10593) | Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros |
| [**Semantic Image Synthesis with Spatially-Adaptive Normalization (SPADE)**](https://arxiv.org/abs/1903.07291) | Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu |
| [**ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models**](https://arxiv.org/abs/2302.05543) | Lvmin Zhang, Maneesh Agrawala |
| [**InstructPix2Pix: Learning to Follow Image Editing Instructions**](https://arxiv.org/abs/2211.09800) | Tim Brooks, Aleksander Holynski, Alexei A. Efros |
| [**Palette: Image-to-Image Diffusion Models**](https://arxiv.org/abs/2111.05826) | Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Rui Zhang, Jason Baldridge, Mohammad Norouzi |
| [**RePaint: Inpainting using Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2201.09865) | Andreas Lugmayr, Martin Danelljan, Luc Van Gool, Radu Timofte |
| [**Diffusion Self-Guidance for Controllable Image Generation**](https://arxiv.org/abs/2305.08891) | Yilun Xu, Zhoutong Zhang, Jiajun Wu |
| [**Diffusion Model-Based Image Editing: A Survey**](https://arxiv.org/abs/2402.17525) | Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, Liangliang Cao |
| [**Controllable Generation with Text-to-Image Diffusion Models: A Survey**](https://arxiv.org/abs/2403.04279) | Yifan Li, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen |

### 4️⃣ Text-to-Video and Video Generation
| **Title** | **Authors** |
|-----------|-------------|
| [**Make-A-Video: Text-to-Video Generation without Text-Video Data**](https://arxiv.org/abs/2209.14792) | Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, Yaniv Taigman |
| [**CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers**](https://arxiv.org/abs/2205.15868) | Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, Jie Tang |
| [**A Survey on Video Diffusion Models**](https://arxiv.org/abs/2310.10647) | Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, Yu-Gang Jiang |
| [**DirecT2V: Large Language Models are Frame-Level Directors for Zero-Shot Text-to-Video Generation**](https://arxiv.org/abs/2305.14330) | Susung Hong, Junyoung Seo, Heeseong Shin, Sunghwan Hong, Seungryong Kim |
| [**VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models**](https://arxiv.org/abs/2401.09047) | Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, Ying Shan |
| [**Sora as an AGI World Model? A Complete Survey on Text-to-Video Generation**](https://arxiv.org/abs/2403.05131) | Joseph Cho, Fachrina Dewi Puspitasari, Sheng Zheng, Jingyao Zheng, Lik-Hang Lee, Tae-Ho Kim, Choong Seon Hong, Chaoning Zhang |
| [**Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation**](https://arxiv.org/abs/2311.17117) | Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, Liefeng Bo |
| [**CameraCtrl: Enabling Camera Control for Text-to-Video Generation**](https://arxiv.org/abs/2404.02101) | Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, Ceyuan Yang |

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
