# ai-image-video-generation-survey

## 🎨 Survey of AI Image and Video Generation Models: Foundations, Taxonomy, and Societal Impact

> *Brendan McCollum*  
> University of Rochester

<h4 align="center">
  📄 <a href="#">Paper (coming soon)</a> &nbsp;|&nbsp;
  🌐 <a href="https://github.com/Brendanmcco/ai-image-video-generation-survey">Project Page</a>
</h4>

---

## 🔍 Overview

This repository accompanies our comprehensive survey on AI-based **image and video generation**, covering the evolution of generative models from GANs to diffusion models and multimodal LLMs. The paper organizes recent advances across five core dimensions:

- Foundational models and generative mechanisms
- A taxonomy across text-to-image, image-to-image, and text-to-video pipelines
- Key architectural and training innovations
- Applications, tools, and evaluation frameworks
- Societal implications and ethical challenges

Rather than a linear timeline, this survey emphasizes **conceptual transitions**, **technical connections across modalities**, and **emerging trends in multimodal generation**.

---

## 📚 Table of Contents

- [🧠 Introduction](#-overview)
- [🔧 Preliminaries & Core Concepts](#-preliminaries--core-concepts)
- [📊 Taxonomy of Generative Models](#-taxonomy-of-generative-models)
- [⚙️ Technical Innovations](#️-technical-innovations)
- [🛠️ Applications & Evaluation](#-applications--evaluation)
- [🌐 Societal Impact & Future Trends](#-societal-impact--future-trends)
- [📚 References](#-references)
- [📌 Citation](#-citation)

---

## 🔧 Preliminaries & Core Concepts

This section reviews fundamental ideas that underpin generative image and video modeling.

| Concept         | Description                                                  |
|----------------|--------------------------------------------------------------|
| Generative Models | Learn data distributions to synthesize new content         |
| Latent Spaces   | Compressed intermediate representations for generation/editing |
| Model Families  | GANs, VAEs, Diffusion Models, Score-Based Methods            |
| Modalities      | Static image vs. temporally coherent video synthesis         |
| Datasets        | LAION-5B, WebVid-10M, ImageNet, MSR-VTT, UCF-101, etc.       |

---

## 📊 Taxonomy of Generative Models

We categorize modern models by **output type** and **architectural strategy**, spanning:

- **Text-to-Image:** DALL·E 2, GLIDE, Stable Diffusion  
- **Image-to-Image:** ControlNet, InstructPix2Pix, RePaint  
- **Text-to-Video:** VideoCrafter2, Make-A-Video, Emu  

| Type               | Examples                                                   |
|--------------------|------------------------------------------------------------|
| Autoregressive      | PixelCNN, VideoGPT                                         |
| GAN-based           | DCGAN, StyleGAN2, AttnGAN                                  |
| Variational (VAEs)  | VQ-VAE, VQ-GAN                                              |
| Diffusion Models    | DDPM, Stable Diffusion, Imagen, SVD                       |
| Score-Based         | Score SDE, EDM                                             |
| Hybrid/Multimodal   | DALL·E 2, Gen-2, Pika, Runway Gen-3                        |
| Neural Fields / 3D  | GIRAFFE, DreamFusion, Sora                                 |

---

## ⚙️ Technical Innovations

### 🎨 Fidelity and Realism
- Metrics: FID, IS, CLIPScore
- Innovations: Cascaded Diffusion, Super-Resolution Decoders

### 🧭 Controllability
- Conditioning on keypoints, layout masks, textual edits (e.g., ControlNet, InstructPix2Pix)

### 🌀 Temporal Consistency
- Spatiotemporal attention, dual-stream architectures, motion diffusion (e.g., Emu, Lumiere)

### ⚡ Efficiency and Scaling
- Fast samplers (DDIM, DiT), Latent Diffusion, Modular training

---

## 🛠️ Applications & Evaluation

- **Creative Media**: Style transfer, anime/manga generation, stylized animation
- **Data Augmentation**: Using synthetic data for downstream vision tasks
- **Interactive Tools**: Real-time editing, controllable character animation
- **Benchmarks**: GenAI-Bench, VBench, EvalCrafter
- **Challenges**: No consensus on video metrics; human evals vs. automated scores

---

## 🌐 Societal Impact & Future Trends

### ⚖️ Ethical Concerns
- **Bias & Representation**: Reinforcing stereotypes in generated outputs
- **Misinformation**: Deepfakes, synthetic propaganda
- **Copyright**: Unclear ownership of AI-generated content

### 🔮 Multimodal and Future Directions
- **Unified Any-to-Any Models**: CoDi, NExT-GPT, Chameleon
- **Token-Level Fusion**: Bridging LLMs and Diffusion in models like Transfusion
- **Interleaved Reasoning & Generation**: OpenLEAF, Video-LLMs

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

### 3️⃣ Image-to-Image and Conditional Generation
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
| **Title** | **Authors** |
|-----------|-------------|
| [**Improved Techniques for Training Score-Based Generative Models**](https://arxiv.org/abs/2006.09011) | Yang Song, Stefano Ermon |
| [**Classifier-Free Diffusion Guidance**](https://openreview.net/pdf?id=qw8AKxfYbI) | Jonathan Ho, Tim Salimans |
| [**Diffusion Models Beat GANs on Image Synthesis**](https://arxiv.org/abs/2105.05233) | Prafulla Dhariwal, Alex Nichol |
| [**Progressive Distillation for Fast Sampling of Diffusion Models**](https://arxiv.org/abs/2202.00512) | Tim Salimans, Jonathan Ho |
| [**LoRA: Low-Rank Adaptation of Large Language Models**](https://arxiv.org/abs/2106.09685) | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen |
| [**DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**](https://arxiv.org/abs/2208.12242) | Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman |
| [**StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets**](https://arxiv.org/abs/2202.00273) | Axel Sauer, Katja Schwarz, Andreas Geiger |
| [**High-Resolution Image Synthesis with Latent Diffusion Models**](https://arxiv.org/abs/2112.10752) | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer |
| [**Score-Based Generative Modeling through Stochastic Differential Equations**](https://arxiv.org/abs/2011.13456) | Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole |

### 6️⃣ Applications and Tools
| **Title** | **Authors** |
|----------|-------------|
| [**DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation**](https://arxiv.org/abs/2412.07589) | Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, Yunhai Tong |
| [**Manga Generation via Layout-Controllable Diffusion**](https://arxiv.org/abs/2412.19303) | Siyu Chen, Dengjie Li, Zenghao Bao, Yao Zhou, Lingfeng Tan, Yujie Zhong, Zheng Zhao |
| [**PromptFix: You Prompt and We Fix the Photo**](https://arxiv.org/abs/2405.16785) | Y. Yu, Z. Zeng, H. Hua, J. Fu, J. Luo |
| [**AIGS: AI-generated Images as Data Sources for Visual Tasks**](https://arxiv.org/abs/2310.01830) | Yongsheng Yu, Ziyun Zeng, Hang Hua, Jianlong Fu, Jiebo Luo |
| [**Animate Anyone: Controllable Image-to-Video Synthesis for Character Animation**](https://arxiv.org/abs/2311.17117) | Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, Liefeng Bo |
| [**Anisora: Exploring the Frontiers of Animation Video Generation in the Sora Era**](https://arxiv.org/abs/2412.10255) | Yudong Jiang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Bingwen Zhu, Xinwen Zhang, Xingyu Zheng, Jixuan Xu, Yue Zhang, Jinlong Hou, Huyang Sun |

### 7️⃣ Evaluation and Ethics
#### Evaluation Metrics and Benchmarks

| **Title** | **Authors** |
|-----------|-------------|
| [**Evaluating Diffusion Models**](https://huggingface.co/docs/diffusers/en/conceptual/evaluation) | Hugging Face Team |
| [**GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation**](https://arxiv.org/abs/2406.13743) | Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, Deva Ramanan |
| [**VBench: Comprehensive Benchmark Suite for Video Generative Models**](https://vchitect.github.io/VBench-project/) | VBench Team |
| [**EvalCrafter: Benchmarking and Evaluating Large Video Generation Models**](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_EvalCrafter_Benchmarking_and_Evaluating_Large_Video_Generation_Models_CVPR_2024_paper.pdf) | Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, Ying Shan, Tencent AI Lab, City University of Hong Kong, University of Macau, The Chinese University of Hong Kong |
| [**CLIP Score vs FID Pareto Curves**](https://wandb.ai/dalle-mini/dalle-mini/reports/CLIP-score-vs-FID-pareto-curves--VmlldzoyMDYyNTAy) | W&B Team |

#### Ethical Considerations

| **Title** | **Authors** |
|-----------|-------------|
| [**Exploring the Ethical Implications of Generative AI: Bias, Deepfakes, and Misinformation**](https://aimresearch.co/council-posts/exploring-the-ethical-implications-of-generative-ai-bias-deepfakes-and-misinformation) | AIM Research Council |
| [**AI Image Generation and Ethical Considerations**](https://uxtbe.medium.com/ai-image-generation-and-ethical-considerations-4b81dfc0b1a9) | Torresburriel Estudio |
| [**The Ethics of AI Art**](https://mediaengagement.org/research/the-ethics-of-ai-art/) | Center for Media Engagement |
| [**Generative AI Takes Stereotypes and Bias From Bad to Worse**](https://www.bloomberg.com/graphics/2023-generative-ai-bias/) | Bloomberg Graphics Team |
| [**OpenAI's Sora Is Plagued by Sexist, Racist, and Ableist Biases**](https://www.wired.com/story/openai-sora-video-generator-bias) | WIRED Staff (2025) |

### 8️⃣ Future Directions and Multimodal Trends
| **Title** | **Authors** |
|-----------|-------------|
| [**Any-to-Any Generation via Composable Diffusion (CoDi)**](https://arxiv.org/abs/2305.11846) | Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal |
| [**CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation**](https://arxiv.org/abs/2311.18775) | Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, Mohit Bansal |
| [**NExT-GPT: Any-to-Any Multimodal LLM**](https://arxiv.org/abs/2309.05519) | Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua |
| [**Chameleon: Mixed-Modal Early-Fusion Foundation Models**](https://arxiv.org/abs/2405.09818) | Chameleon Team |
| [**Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model**](https://arxiv.org/abs/2408.11039) | Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy |
| [**OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation**](https://arxiv.org/abs/2310.07749) | Jie An, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Lijuan Wang, Jiebo Luo |
| [**A Survey on Multimodal Large Language Models**](https://arxiv.org/abs/2306.13549) | Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, Enhong Chen |
| [**A Survey on Video Diffusion Models**](https://arxiv.org/abs/2310.10647) | Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, Yu-Gang Jiang |

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
