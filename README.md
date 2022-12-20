# ECCV-2022-Papers
![9361c19ba6cbc5ae7be1fba8d82759b](https://user-images.githubusercontent.com/62801906/150054267-d54c28a4-f2de-4171-88df-d9820222a392.jpg)

官网链接：https://eccv2022.ecva.net/

截稿日期：2022年3月7日(9:59PM CET, 11:59AM PST)

会议日期：2022年10月24日-2022年10月28日

## 历年综述论文分类汇总戳这里↘️[CV-Surveys](https://github.com/52CV/CV-Surveys)施工中~~~~~~~~~~

## 2022 年论文分类汇总戳这里
↘️[CVPR-2022-Papers](https://github.com/52CV/CVPR-2022-Papers)
↘️[WACV-2022-Papers](https://github.com/52CV/WACV-2022-Papers)
↘️[ECCV-2022-Papers](https://github.com/52CV/ECCV-2022-Papers)

## 2021年论文分类汇总戳这里
↘️[ICCV-2021-Papers](https://github.com/52CV/ICCV-2021-Papers)
↘️[CVPR-2021-Papers](https://github.com/52CV/CVPR-2021-Papers)

## 2020 年论文分类汇总戳这里
↘️[CVPR-2020-Papers](https://github.com/52CV/CVPR-2020-Papers)
↘️[ECCV-2020-Papers](https://github.com/52CV/ECCV-2020-Papers)

## ❣❣❣另外打包下载ECCV 2022论文，可在【我爱计算机视觉】微信公众号后台回复“paper”。共计 1645 篇。分类进行中......

|:cat:|:dog:|:tiger:|:wolf:|
|------|------|------|------|
|[1.其它](#1)|[2.Image Segmentation(图像分割)](#2)|[3.Image Progress(图像处理)](#3)|[4.Image Captioning(图像字幕)](#4)|
|[5.Image/Video Retrieval(图像/视频检索)](#5)|[6.Object Detection(目标检测)](#6)|[7.Object Tracking(目标跟踪)](#7)|[8.3D(三维视觉)](#8)|
|[9.Human Pose Estimation(人体姿态估计)](#9)|[10.Pose Estimation(物体姿势估计)](#10)|[11.Video](#11)|[12.Action Detection(人体动作检测与识别)](#12)|
|[13.Human-Object Interaction(人物交互)](#13)|[14.Visual Answer Questions(视觉问答)](#14)|[15.Vision-Language(视觉语言)](#15)|[16.Transformer](#16)|
|[17.GAN](#17)|[18.Image-to-Image Translation(图像到图像翻译)](#18)|[19.Image Synthesis/Generation(图像合成)](#19)|[20.Face(人脸)](#20)|
|[21.Semi/self-supervised learning(半/自监督)](#21)|[22.OCR](#22)|[23.Medical Image(医学影像)](#23)|[24.UAV/Remote Sensing/Satellite Image(无人机/遥感/卫星图像)](#24)|

## 12月20日整理 30 篇

* 分割
  * [Unsupervised Segmentation in Real-World Images via Spelke Object Inference](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890708.pdf)
  
  * [Fast Two-View Motion Segmentation Using Christoffel Polynomials](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900001.pdf)
  
  * [A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-Language Model](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890725.pdf)<br>:star:[code](https://github.com/MendelXu/zsseg.baseline)
  * [UCTNet: Uncertainty-Aware Cross-Modal Transformer Network for Indoor RGB-D Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900020.pdf)
  * [Cross-Domain Few-Shot Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900072.pdf)<br>:star:[code](https://github.com/slei109/PATNet)
  * [CP2: Copy-Paste Contrastive Pretraining for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900494.pdf)<br>:star:[code](https://github.com/wangf3014/CP2)
  * [HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900370.pdf)<br>:star:[code](https://github.com/lhoyer/HRDA)
  
  * [Learning Regional Purity for Instance Segmentation on 3D Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900055.pdf)

  * [Improving Few-Shot Part Segmentation Using Coarse Supervision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900282.pdf)
* 对比学习
  * [Generative Subgraph Contrast for Self-Supervised Graph Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900090.pdf)<br>:star:[code](https://github.com/yh-han/GSC)
* 无监督
  * [Dense Siamese Network for Dense Unsupervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900460.pdf)<br>:star:[code](https://github.com/ZwwWayne/DenseSiam)
* 半监督
  * [Vibration-Based Uncertainty Estimation for Learning from Limited Supervision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900160.pdf)
  * [Unsupervised Selective Labeling for More Effective Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf)
  * [RDA: Reciprocal Distribution Alignment for Robust Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900527.pdf)<br>:star:[code](https://github.com/NJUyued/RDA4RobustSSL)
  * [Semi-Supervised Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900596.pdf)<br>:star:[code](https://github.com/wengzejia1/Semiformer)
  * [CA-SSL: Class-Agnostic Semi-Supervised Learning for Detection and Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910057.pdf)
* 自监督
  * [Mc-BEiT: Multi-Choice Discretization for Image BERT Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900229.pdf)<br>:star:[code](https://github.com/lixiaotong97/mc-BEiT)
  * [What to Hide from Your Students: Attention-Guided Masked Image Modeling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900299.pdf)<br>:star:[code](https://github.com/gkakogeorgiou/attmask)
  * [Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910021.pdf)<br>:star:[code](https://github.com/UCDvision/CMSF)
  * [Semantic-Aware Fine-Grained Correspondence](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910093.pdf)
  * [Self-Supervised Classification Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910112.pdf)<br>:star:[code](https://github.com/elad-amrani/self-classifier)
  * [Dual-Domain Self-Supervised Learning and Model Adaption for Deep Compressive Imaging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900406.pdf)
* OD
  * [Weakly Supervised Object Localization through Inter-class Feature Similarity and Intra-Class Appearance Consistency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900194.pdf)
  * [Diverse Learner: Exploring Diverse Supervision for Semi-Supervised Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900631.pdf)
  * [W2N: Switching from Weak Supervision to Noisy Supervision for Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900699.pdf)<br>:star:[code](https://github.com/1170300714/w2n_wsod)
* AD
  * [Locally Varying Distance Transform for Unsupervised Visual Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900353.pdf)
  * [SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf)<br>:star:[code](https://github.com/amazon-science/spot-diff)
* 去模糊
  * [United Defocus Blur Detection and Deblurring via Adversarial Promoting Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900562.pdf)<br>:star:[code](https://github.com/wdzhao123/APL)
* 其它
  * [MVP: Multimodality-Guided Visual Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900336.pdf)
  * [Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900511.pdf)



### —————————————————————————————————————————————
* human relighting
  * [Geometry-aware Single-image Full-body Human Relighting](https://arxiv.org/abs/2207.04750)
* 奇异值检测(Novelty Detection)
  * [Semantic Novelty Detection via Relational Reasoning](https://arxiv.org/abs/2207.08699)<br>:star:[code](https://github.com/FrancescoCappio/ReSeND)
* Multi-attribute Learning
  * [Label2Label: A Language Modeling Framework for Multi-Attribute Learning](https://arxiv.org/abs/2207.08677)<br>:star:[code](https://github.com/Li-Wanhua/Label2Label)
* 偏见识别
  * [Discover and Mitigate Unknown Biases with Debiasing Alternate Networks](https://arxiv.org/abs/2207.10077)<br>:star:[code](https://github.com/zhihengli-UR/DebiAN) 
* 新类别发现(Novel Class Discovery)
  * [Novel Class Discovery without Forgetting](https://arxiv.org/abs/2207.10659)
* 密集预测
  * [FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling](https://arxiv.org/abs/2207.10392)<br>:star:[code](https://github.com/poppinace/fade)
* 变分自动编码器(VAEs) 
  * [Continual Variational Autoencoder Learning via Online Cooperative Memorization](https://arxiv.org/abs/2207.10131)<br>:star:[code](https://github.com/dtuzi123/OVAE)
* 开集识别
  * [Towards Accurate Open-Set Recognition via Background-Class Regularization](https://arxiv.org/abs/2207.10287)
* 草图
  * [Abstracting Sketches through Simple Primitives](https://arxiv.org/abs/2207.13543)<br>:star:[code](https://github.com/ExplainableML/sketch-primitives)
  * [FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680245.pdf)<br>:house:[project](https://fscoco.github.io/)
  * [DoodleFormer: Creative Sketch Drawing with Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770343.pdf)<br>:star:[code](https://github.com/ankanbhunia/doodleformer) 
* 聚类
  * [A Data-Centric Approach for Improving Ambiguous Labels with Combined Semi-Supervised Classification and Clustering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680354.pdf)<br>:star:[code](https://github.com/Emprime/dc3) 
* Visual Grounding
  * [SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding](https://arxiv.org/abs/2207.13325)<br>:star:[code](https://github.com/qumengxue/siri-vg)
* 互动结构理解
  * [Break and Make: Interactive Structural Understanding Using LEGO Bricks](https://arxiv.org/abs/2207.13738)<br>:star:[code](https://github.com/aaronwalsman/ltron)
* HDR全景图生成
  * [StyleLight: HDR Panorama Generation for Lighting Estimation and Editing](https://arxiv.org/abs/2207.14811)<br>:star:[code](https://github.com/Wanggcong/StyleLight):house:[project](https://style-light.github.io/)
* 手语识别
  * [Automatic dense annotation of large-vocabulary sign language videos](https://arxiv.org/abs/2208.02802)<br>:house:[project](https://www.robots.ox.ac.uk/~vgg/research/bsldensify/)
  * [Deep Radial Embedding for Visual Sequence Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660234.pdf)   
* 读唇术
  * [Speaker-adaptive Lip Reading with User-dependent Padding](https://arxiv.org/abs/2208.04498)
* BNN
  * [Recurrent Bilinear Optimization for Binary Neural Networks](https://arxiv.org/abs/2209.01542)<br>:open_mouth:oral:star:[code](https://github.com/SteveTsui/RBONN)
  * [Towards Accurate Binary Neural Networks via Modeling Contextual Dependencies](https://arxiv.org/abs/2209.01404)<br>:star:[code](https://github.com/Sense-GVT/BCDNet)
* 图像匹配
  * [ECO-TR: Efficient Correspondences Finding Via Coarse-to-Fine Refinement](https://arxiv.org/abs/2209.12213)<br>:star:[code](https://github.com/dltan7/ECO-TR):house:[project](https://dltan7.github.io/ecotr/)
* 图像取证
  * [Totems: Physical Objects for Verifying Visual Integrity](https://arxiv.org/abs/2209.13032)<br>:house:[project](https://jingweim.github.io/totems/)
* 图像对齐
  * [Learning Depth from Focus in the Wild](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610001.pdf)<br>:star:[code](https://github.com/wcy199705/DfFintheWild)  
* visual hand pressure estimation
  * [PressureVision: Estimating Hand Pressure from a Single RGB Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660322.pdf)<br>:star:[code](https://github.com/facebookresearch/pressurevision)  
* 光亮估计
  * [Estimating Spatially-Varying Lighting in Urban Scenes with Disentangled Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660445.pdf)
* 室内场景照明编辑
  * [Physically-Based Editing of Indoor Scene Lighting from a Single Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660545.pdf)
* HDR
  * [Exposure-Aware Dynamic Weighted Learning for Single-Shot HDR Imaging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670429.pdf)<br>:star:[code](https://github.com/viengiaan/EDWL) 
* 关键点定位
  * [Multi-Domain Multi-Definition Landmark Localization for Small Datasets](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690637.pdf) 
* XAI
  * [STEEX: Steering Counterfactual Explanations with Semantics](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720382.pdf)<br>:star:[code](https://github.com/valeoai/STEEX)
  * [Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720260.pdf)<br>:star:[code](https://github.com/facebookresearch/visual-counterfactuals)
  * [HIVE: Evaluating the Human Interpretability of Visual Explanations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720277.pdf)<br>:house:[project](https://princetonvisualai.github.io/HIVE)
  * [Shap-CAM: Visual Explanations for Convolutional Neural Networks Based on Shapley Value](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720455.pdf)
* 掌纹识别
  * [BézierPalm: A Free Lunch for Palmprint Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730019.pdf)<br>:house:[project](https://kaizhao.net/palmprint) 
* 视线估计
  * [Look Both Ways: Self-Supervising Driver Gaze Estimation and Road Scene Saliency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730128.pdf)<br>:open_mouth:oral:star:[code](https://github.com/Kasai2020/look_both_ways)
* 运动迁移
  * [Motion and Appearance Adaptation for Cross-Domain Motion Transfer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760506.pdf)


## 光学、几何、光场成像
* 相机相关
  * [Learned Monocular Depth Priors in Visual-Inertial Initialization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820537.pdf) 
  * 相机姿态估计
    * [E-Graph: Minimal Solution for Rigid Rotation with Extensibility Graphs](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820298.pdf)
  * 相机姿势
    * [Camera Pose Auto-Encoders for Improving Pose Regression](https://arxiv.org/abs/2207.05530)<br>:star:[code](https://github.com/yolish/camera-pose-auto-encoders)
  * 相机估计
    * [A Reliable Online Method for Joint Estimation of Focal Length and Camera Rotation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610247.pdf)<br>:star:[code](https://github.com/ElderLab-York-University/OnlinefR)
  * 相机自动校准
    * [Camera Auto-Calibration from the Steiner Conic of the Fundamental Matrix](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620419.pdf) 
  * 事件相机
    * [DVS-Voltmeter: Stochastic Process-Based Event Simulator for Dynamic Vision Sensors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670571.pdf)<br>:star:[code](https://github.com/Lynn0306/DVS-Voltmeter)  
  * 相机重识别
    * [SC-wLS: Towards Interpretable Feed-forward Camera Re-localization](https://arxiv.org/abs/2210.12748)<br>:star:[code](https://github.com/XinWu98/SC-wLS)
  * 相机定位
    * [Towards Accurate Active Camera Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700119.pdf)<br>:star:[code](https://github.com/qhFang/AccurateACL)
* 光场
  * [Neural Light Field Estimation for Street Scenes with Differentiable Virtual Object Insertion](https://arxiv.org/abs/2208.09480)<br>:house:[project](https://nv-tlabs.github.io/outdoor-ar/)
  * [Synthesizing Light Field Video from Monocular Video](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670158.pdf)<br>:star:[code](https://github.com/ShrisudhanG/Synthesizing-Light-Field-Video-from-Monocular-Video)

## Data Augmentation(数据增强)
* [TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers](https://arxiv.org/abs/2207.08409)<br>:star:[code](https://github.com/Sense-X/TokenMix)
* [Neuromorphic Data Augmentation for Training Spiking Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670623.pdf)
* [3D Random Occlusion and Multi-layer Projection for Deep Multi-Camera Pedestrian Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700681.pdf)<br>:star:[code](https://github.com/xjtlu-cvlab/3DROM)

## Image Matching(图像匹配)
* [ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer](https://arxiv.org/abs/2208.14201)<br>:house:[project](https://aspanformer.github.io/)

## 人体动作预测
* [ERA: Expert Retrieval and Assembly for Early Action Prediction](https://arxiv.org/abs/2207.09675)
* [Overlooked Poses Actually Make Sense: Distilling Privileged Knowledge for Human Motion Prediction](https://arxiv.org/abs/2208.01302)
* [GIMO: Gaze-Informed Human Motion Prediction in Context](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730675.pdf)<br>:star:[code](https://github.com/y-zheng18/GIMO) 
* [Diverse Human Motion Prediction Guided by Multi-level Spatial-Temporal Anchors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820244.pdf)<br>:star:[code](https://github.com/Sirui-Xu/STARS)
* 行动预测
  * [Rethinking Learning Approaches for Long-Term Action Anticipation](https://arxiv.org/abs/2210.11566)<br>:star:[code](https://github.com/Nmegha2601/anticipatr)
* 运动估计
  * [PREF: Predictability Regularized Neural Motion Fields](https://arxiv.org/abs/2209.10691)<br>:open_mouth:oral
* 人体运动合成
  * [Learning Uncoupled-Modulation CVAE for 3D Action-Conditioned Human Motion Synthesis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810707.pdf)
  * [MotionCLIP: Exposing Human Motion Generation to CLIP Space](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820349.pdf)<br>:star:[code](https://guytevet.github.io/motionclip-page/)

## Scene Graph Generation(场景图生成)
* [Panoptic Scene Graph Generation](https://arxiv.org/abs/2207.11247)<br>:star:[code](https://github.com/Jingkang50/OpenPSG/):house:[project](https://psgdataset.org/)
* [Meta Spatio-Temporal Debiasing for Video Scene Graph Generation](https://arxiv.org/abs/2207.11441)
* [Hierarchical Memory Learning for Fine-Grained Scene Graph Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870263.pdf)
* [Fine-Grained Scene Graph Generation with Data Transfer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870402.pdf)<br>:star:[code](https://github.com/waxnkw/IETrans-SGG.pytorch)
* [Towards Open-Vocabulary Scene Graph Generation with Prompt-Based Finetuning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880055.pdf)  
  
## Sound
* 视听分割
  * [Audio-Visual Segmentation](https://arxiv.org/abs/2207.05042)<br>:star:[code](https://github.com/OpenNLPLab/AVSBench)
* 语音合成
  * [VisageSynTalk: Unseen Speaker Video-to-Speech Synthesis via Speech-Visage Feature Selection](https://arxiv.org/abs/2206.07458)
* 声音分离
  * [AudioScopeV2: Audio-Visual Attention Architectures for Calibrated Open-Domain On-Screen Sound Separation](https://arxiv.org/abs/2207.10141)
 

## Style Transfer(风格迁移)
* [CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer](https://arxiv.org/abs/2207.04808)<br>:open_mouth:oral:star:[code](https://github.com/JarrentWu1031/CCPL)
* [Learning Graph Neural Networks for Image Style Transfer](https://arxiv.org/abs/2207.11681)
* 图像风格化
  * [WISE: Whitebox Image Stylization by Example-based Learning](https://arxiv.org/abs/2207.14606)<br>:star:[code](https://github.com/winfried-loetzsch/wise)
* 发型迁移
  * [Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment](https://arxiv.org/abs/2208.07765)<br>:star:[code](https://github.com/Taeu/Style-Your-Hair)

## View Generation(视图生成)
* [InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images](https://arxiv.org/abs/2207.11148)<br>:open_mouth:oral
* [CompNVS: Novel View Synthesis with Scene Completion](https://arxiv.org/abs/2207.11467)
* [HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields](https://arxiv.org/abs/2208.06787)<br>:star:[code](https://github.com/postech-ami/HDR-Plenoxels)
* [Neural Radiance Transfer Fields for Relightable Novel-View Synthesis with Global Illumination](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770155.pdf) 

## Dataset(数据集)
* [COO: Comic Onomatopoeia Dataset for Recognizing Arbitrary or Truncated Texts](https://arxiv.org/abs/2207.04675)<br>:star:[code](https://github.com/ku21fan/COO-Comic-Onomatopoeia)<br>用于识别任意或截断文本的漫画拟声词数据集
* [BRACE: The Breakdancing Competition Dataset for Dance Motion Synthesis](https://arxiv.org/abs/2207.10120)<br>:sunflower:[dataset](https://github.com/dmoltisanti/brace/)<br>用于舞蹈动作合成的霹雳舞比赛数据集
* [CelebV-HQ: A Large-Scale Video Facial Attributes Dataset](https://arxiv.org/abs/2207.12393)<br>:sunflower:[dataset](https://github.com/CelebV-HQ/CelebV-HQ):house:[project](https://celebv-hq.github.io/)<br>一个大规模的视频人脸属性数据集
* [UnrealEgo: A New Dataset for Robust Egocentric 3D Human Motion Capture](https://arxiv.org/abs/2208.01633)<br>:star:[code](https://github.com/hiroyasuakada/UnrealEgo):house:[project](https://4dqv.mpi-inf.mpg.de/UnrealEgo/)<br>用于鲁棒性以自我为中心的三维人类运动捕捉的新数据集
* [BEAT: A Large-Scale Semantic and Emotional Multi-modal Dataset for Conversational Gestures Synthesis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670605.pdf)<br>:sunflower:[dataset](https://pantomatrix.github.io/BEAT/)<br>会话手势合成
* [MovieCuts: A New Dataset and Benchmark for Cut Type Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670659.pdf)<br>:sunflower:[dataset](https://github.com/PardoAlejo/MovieCuts)<br>剪切类型识别  
* [A Real World Dataset for Multi-View 3D Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680054.pdf)<br>:sunflower:[dataset](http://www.ocrtoc.org/#/3D-Reconstruction)<br>三维重建
* [Capturing, Reconstructing, and Simulating: The UrbanScene3D Dataset](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680090.pdf)<br>:sunflower:[dataset](https://vcc.tech/UrbanScene3D)<br>城市场景重建
* [PartImageNet: A Large, High-Quality Dataset of Parts](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680124.pdf)<br>分割
* [A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680141.pdf)<br>:sunflower:[dataset](https://allenai.org/project/a-okvqa/home)<br>VQA
* [OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680158.pdf)
* [The Anatomy of Video Editing: A Dataset and Benchmark Suite for AI-Assisted Video Editing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680195.pdf)<br>:sunflower:[dataset](https://github.com/dawitmureja/AVE)<br>视频编辑
* [ClearPose: Large-Scale Transparent Object Dataset and Benchmark](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680372.pdf)<br>:sunflower:[dataset](https://github.com/opipari/ClearPose)<br>深度估计
* [AnimeCeleb: Large-Scale Animation CelebHeads Dataset for Head Reenactment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680405.pdf)<br>:sunflower:[dataset](https://github.com/kangyeolk/AnimeCeleb)<br>动画名人头像数据集
* [A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680440.pdf)<br>用于室内和室外场景解析的密集材料分割数据集
* [MimicME: A Large Scale Diverse 4D Database for Facial Expression Analysis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680457.pdf)<br>用于面部表情分析的大规模多样化4D数据库
* [Delving into Universal Lesion Segmentation: Method, Dataset, and Benchmark](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680475.pdf)<br>:sunflower:[dataset](https://github.com/yuqiuyuqiu/KEN)<br>病变分割  

## Scene Flow Estimation(场景流估计)
* [Bi-PointFlowNet: Bidirectional Learning for Point Cloud Based Scene Flow Estimation](https://arxiv.org/abs/2207.07522)<br>:star:[code](https://github.com/cwc1260/BiFlow)
* [What Matters for 3D Scene Flow Network](https://arxiv.org/abs/2207.09143)<br>:star:[code](https://github.com/IRMVLab/3DFlow)
* [MonoPLFlowNet: Permutohedral Lattice FlowNet for Real-Scale 3D Scene Flow Estimation with Monocular Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870316.pdf)  

## Anomaly Detection(异常检测)
* [Registration based Few-Shot Anomaly Detection](https://arxiv.org/abs/2207.07361)<br>:open_mouth:oral:star:[code](https://github.com/MediaBrain-SJTU/RegAD)
* [Dynamic Local Aggregation Network with Adaptive Clusterer for Anomaly Detection](https://arxiv.org/abs/2207.10948)<br>:star:[code](https://github.com/Beyond-Zw/DLAN-AC)
* [Locally Varying Distance Transform for Unsupervised Visual Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900353.pdf)
* [SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900389.pdf)<br>:star:[code](https://github.com/amazon-science/spot-diff)
* [HaloAE: An HaloNet based Local Transformer Auto-Encoder for Anomaly Detection and Localization](https://arxiv.org/abs/2208.03486)
* [Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850107.pdf)<br>:star:[code](https://github.com/GaoangW/HSCL)
* 表面异常检测
  * [DSR -- A dual subspace re-projection network for surface anomaly detection](https://arxiv.org/abs/2208.01521)<br>:star:[code](https://github.com/VitjanZ/DSR_anomaly_detection)

## 渲染
* [Relighting4D: Neural Relightable Human from Videos](https://arxiv.org/abs/2207.07104)<br>:star:[code](https://github.com/FrozenBurning/Relighting4D):house:[project](https://frozenburning.github.io/projects/relighting4d/):tv:[video](https://www.youtube.com/watch?v=NayAw89qtsY)
* [MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects](https://arxiv.org/abs/2207.08403)<br>:star:[code](https://github.com/JuewenPeng/MPIB):house:[project](https://juewenpeng.github.io/MPIB/) 
* [Approximate Differentiable Rendering with Algebraic Surfaces](https://arxiv.org/abs/2207.10606)<br>:star:[code](https://github.com/leonidk/fuzzy-metaballs):house:[project](https://leonidk.github.io/fuzzy-metaballs/)
* [AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields](https://arxiv.org/abs/2207.10312)<br>:star:[code](https://github.com/thomasneff/AdaNeRF):house:[project](https://thomasneff.github.io/adanerf/) 
* [Generalizable Patch-Based Neural Rendering](https://arxiv.org/abs/2207.10662)<br>:open_mouth:oral:star:[code](https://github.com/google-research/google-research/tree/master/gen_patch_neural_rendering):house:[project](https://mohammedsuhail.net/gen_patch_neural_rendering/)
* [Deforming Radiance Fields with Cages](https://arxiv.org/abs/2207.12298)<br>:star:[code](https://github.com/xth430/deforming-nerf):house:[project](https://xth430.github.io/deforming-nerf/)
* [NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing](https://arxiv.org/abs/2207.11911)<br>:open_mouth:oral:star:[code](https://github.com/zju3dv/neumesh):house:[project](https://zju3dv.github.io/neumesh/)  
* [ActiveNeRF: Learning where to See with Uncertainty Estimation](https://arxiv.org/abs/2209.08546)<br>:star:[code](https://github.com/LeapLabTHU/ActiveNeRF)
* [ARAH: Animatable Volume Rendering of Articulated Human SDFs](https://arxiv.org/abs/2210.10036)<br>:star:[code](https://github.com/taconite/arah-release):house:[project](https://neuralbodies.github.io/arah/)
* [LaTeRF: Label and Text Driven Object Radiance Fields](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630021.pdf)
* [MoFaNeRF: Morphable Facial Neural Radiance Field](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630267.pdf)<br>:star:[code](https://github.com/zhuhao-nju/mofanerf)
* [Conditional-Flow NeRF: Accurate 3D Modelling with Reliable Uncertainty Quantification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630531.pdf) 
* [Sem2NeRF: Converting Single-View Semantic Masks to Neural Radiance Fields](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740713.pdf)<br>:star:[code](https://donydchen.github.io/sem2nerf/)
* [KeypointNeRF: Generalizing Image-Based Volumetric Avatars Using Relative Spatial Encoding of Keypoints](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750176.pdf)<br>:house:[project](https://markomih.github.io/KeypointNeRF)
* [ViewFormer: NeRF-Free Neural Rendering from Few Images Using Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750195.pdf)<br>:star:[code](https://github.com/jkulhanek/viewformer)
* [GeoAug: Data Augmentation for Few-Shot NeRF with Geometry Constraints](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770326.pdf)
* [SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820712.pdf)<br>:star:[code](https://vita-group.github.io/SinNeRF/)  

## Few/Zero-Shot Learning/Domain Generalization/Adaptation(小/零样本/域泛化/适应)
* 小样本
  * [Cross-Domain Cross-Set Few-Shot Learning via Learning Compact and Aligned Representations](https://arxiv.org/abs/2207.07826)<br>:star:[code](https://github.com/WentaoChen0813/CDCS-FSL)
  * [Self-Supervision Can Be a Good Few-Shot Learner](https://arxiv.org/abs/2207.09176)<br>:star:[code](https://github.com/bbbdylan/unisiam)
  * [VizWiz-FewShot: Locating Objects in Images Taken by People With Visual Impairments](https://arxiv.org/abs/2207.11810)<br>:house:[project](https://vizwiz.org/)
  * [Contrastive Prototypical Network with Wasserstein Confidence Penalty](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790654.pdf)<br>:star:[code](https://github.com/Haoqing-Wang/CPNWCP)
  * [tSF: Transformer-Based Semantic Filter for Few-Shot Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800001.pdf)
  * [Worst Case Matters for Few-Shot Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800092.pdf)
  * [Learning Instance and Task-Aware Dynamic Kernels for Few-Shot Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800247.pdf)
  * [Self-Promoted Supervision for Few-Shot Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800318.pdf)<br>:star:[code](https://github.com/DongSky/few-shot-vit)
  * [Improving Few-Shot Learning through Multi-task Representation Learning Theory](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800423.pdf)
  * [TransVLAD: Focusing on Locally Aggregated Descriptors for Few-Shot Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800509.pdf)
  * [Kernel Relative-Prototype Spectral Filtering for Few-Shot Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800527.pdf)<br>:star:[code](https://github.com/zhangtao2022/DSFN)
  * [Uncertainty-DTW for Time Series and Sequences](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810174.pdf)
* 零样本
  * [Temporal and cross-modal attention for audio-visual zero-shot learning](https://arxiv.org/abs/2207.09966)<br>:star:[code](https://github.com/ExplainableML/TCAF-GZSL)
  * [3D Compositional Zero-Shot Learning with DeCompositional Consensus](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880704.pdf)
  * [Learning Invariant Visual Representations for Compositional Zero-Shot Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840335.pdf) <br>:star:[code](https://github.com/PRIS-CV/IVR)
* 域适应
  * [Prior Knowledge Guided Unsupervised Domain Adaptation](https://arxiv.org/abs/2207.08877)<br>:star:[code](https://github.com/tsun/KUDA)
  * [CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation](https://arxiv.org/abs/2207.09778)<br>:star:[code](https://github.com/saltoricristiano/cosmix-uda)
  * [GIPSO: Geometrically Informed Propagation for Online Adaptation in 3D LiDAR Segmentation](https://arxiv.org/abs/2207.09763)<br>:star:[code](https://github.com/saltoricristiano/gipso-sfouda)
  * [Prototype-Guided Continual Adaptation for Class-Incremental Unsupervised Domain Adaptation](https://arxiv.org/abs/2207.10856)<br>:star:[code](https://github.com/Hongbin98/ProCA)
  * [MemSAC: Memory Augmented Sample Consistency for Large Scale Domain Adaptation](https://arxiv.org/abs/2207.12389)<br>:star:[code](https://github.com/ViLab-UCSD/MemSAC_ECCV2022):house:[project](https://tarun005.github.io/MemSAC/)
  * [Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation](https://arxiv.org/abs/2207.13247)<br>:star:[code](https://github.com/val-iisc/StickerDA):house:[project](https://sites.google.com/view/sticker-sfda)
  * [Combating Label Distribution Shift for Active Domain Adaptation](https://arxiv.org/abs/2208.06604)
  * [Uncertainty-guided Source-free Domain Adaptation](https://arxiv.org/abs/2208.07591)<br>:star:[code](https://github.com/roysubhankar/uncertainty-sfda)
* 域泛化
  * [Grounding Visual Representations with Texts for Domain Generalization](https://arxiv.org/abs/2207.10285)<br>:star:[code](https://github.com/mswzeus/GVRT)
  * [Improving Test-Time Adaptation via Shift-agnostic Weight Regularization and Nearest Source Prototypes](https://arxiv.org/abs/2207.11707)
  * [Attention Diversification for Domain Generalization](https://arxiv.org/abs/2210.04206)<br>:star:[code](https://github.com/hikvision-research/DomainGeneralization)
  * [Cross-Domain Ensemble Distillation for Domain Generalization](https://arxiv.org/abs/2211.14058)
  * [Domain Generalization by Mutual-Information Regularization with Pre-trained Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830427.pdf)<br>:star:[code](https://github.com/kakaobrain/miro)
  * [MVDG: A Unified Multi-View Framework for Domain Generalization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870158.pdf)<br>:star:[code](https://github.com/koncle/MVDG) 

## Semantic Correspondence(语义对应)
* [Demystifying Unsupervised Semantic Correspondence Estimation](https://arxiv.org/abs/2207.05054)<br>:star:[code](https://github.com/MehmetAygun/demistfy_correspondence):house:[project](https://mehmetaygun.github.io/demistfy.html)
* [Learning Semantic Correspondence with Sparse Annotations](https://arxiv.org/abs/2208.06974)

## GNN/GCN(图神经网络)
* GCN
  * [End-to-end Graph-constrained Vectorized Floorplan Generation with Panoptic Refinement](https://arxiv.org/abs/2207.13268) 
  * [Learning Self-Prior for Mesh Denoising Using Dual Graph Convolutional Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630358.pdf)<br>:star:[code](https://github.com/astaka-pe/Dual-DMP)
* GNN
  * [Adversarial Label Poisoning Attack on Graph Neural Networks via Label Propagation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650223.pdf)
  * [Equivariant Hypergraph Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810086.pdf)<br>:star:[code](https://github.com/jw9730/ehnn) 
  
## Continual Learning(持续学习)
* [Balancing Stability and Plasticity through Advanced Null Space in Continual Learning](https://arxiv.org/abs/2207.12061)<br>:open_mouth:oral 
* [Online Continual Learning with Contrastive Vision Transformer](https://arxiv.org/abs/2207.13516) 
* [Helpful or Harmful: Inter-Task Association in Continual Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710518.pdf)
* [Theoretical Understanding of the Information Flow on Continual Learning Performance](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720085.pdf)<br>:star:[code](https://github.com/Sekeh-Lab/InformationFlow-CL)
* [Transfer without Forgetting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830672.pdf)<br>:star:[code](https://github.com/mbosc/twf)
* [incDFM: Incremental Deep Feature Modeling for Continual Novelty Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850581.pdf)
* [Online Task-Free Continual Learning with Dynamic Sparse Distributed Memory](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850721.pdf)<br>:star:[code](https://github.com/Julien-pour/Dynamic-Sparse-Distributed-Memory)
* [Discriminability-Transferability Trade-Off: An Information-Theoretic Perspective](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860020.pdf)
* [CoSCL: Cooperation of Small Continual Learners Is Stronger than a Big One](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860249.pdf)<br>:star:[code](https://github.com/lywang3081/CoSCL) 
* [DualPrompt: Complementary Prompting for Rehearsal-Free Continual Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860617.pdf)<br>:star:[code](https://github.com/google-research/l2p) 

## Metric Learning(度量学习)
* [DAS: Densely-Anchored Sampling for Deep Metric Learning](https://arxiv.org/abs/2208.00119)<br>:star:[code](https://github.com/lizhaoliu-Lec/DAS)
* [Posterior Refinement on Metric Matrix Improves Generalization Bound in Metric Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860199.pdf)
* [A Non-Isotropic Probabilistic Take On Proxy-Based Deep Metric Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf)<br>:star:[code](https://github.com/ExplainableML/Probabilistic_Deep_Metric_Learning)
## Active Learning(主动学习)
* [When Active Learning Meets Implicit Semantic Data Augmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850056.pdf)
* [PT4AL: Using Self-Supervised Pretext Tasks for Active Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860583.pdf)<br>:star:[code](https://github.com/johnsk95/PT4AL)  

## Lifelong Learning(终生学习)
* [Anti-Retroactive Interference for Lifelong Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840160.pdf)<br>:star:[code](https://github.com/bhrqw/ARI) 

## Reinforcement Learning(强化学习)
* [Style-Agnostic Reinforcement Learning](https://arxiv.org/abs/2208.14863)<br>:star:[code](https://github.com/POSTECH-CVLab/style-agnostic-RL)

## Incremental Learning(增量学习)
* [Learning with Recoverable Forgetting](https://arxiv.org/abs/2207.08224)
* [Incremental Task Learning with Incremental Rank Updates](https://arxiv.org/abs/2207.09074)<br>:star:[code](https://github.com/CSIPlab/task-increment-rank-update)
* [DLCFT: Deep Linear Continual Fine-Tuning for General Incremental Learning](https://arxiv.org/abs/2208.08112)
* 类增量
  * [Class-incremental Novel Class Discovery](https://arxiv.org/abs/2207.08605)<br>:star:[code](https://github.com/OatmealLiu/class-iNCD) 
  * [Few-Shot Class-Incremental Learning via Entropy-Regularized Data-Free Replay](https://arxiv.org/abs/2207.11213)
  * [Few-Shot Class-Incremental Learning from an Open-Set Perspective](https://arxiv.org/abs/2208.00147)<br>:star:[code](https://github.com/CanPeng123/FSCIL_ALICE)
  * [Class-Incremental Learning with Cross-Space Clustering and Controlled Transfer](https://arxiv.org/abs/2208.03767)<br>:star:[code](https://github.com/richzhang/webpage-template):house:[project](https://cscct.github.io/)
  * [R-DFCIL: Relation-Guided Representation Learning for Data-Free Class Incremental Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830411.pdf)<br>:star:[code](https://github.com/jianzhangcs/R-DFCIL)
  * [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850393.pdf)
  * [S3C: Self-Supervised Stochastic Classifiers for Few-Shot Class-Incremental Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850427.pdf)<br>:star:[code](https://github.com/JAYATEJAK/S3C)

## Adversarial  Learning(对抗学习)
* [Prior-Guided Adversarial Initialization for Fast Adversarial Training](https://arxiv.org/abs/2207.08859)<br>:star:[code](https://github.com/jiaxiaojunQAQ/FGSM-PGI)
* [BIPS: Bi-modal Indoor Panorama Synthesis via Residual Depth-Aided Adversarial Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760331.pdf)<br>:star:[code](https://github.com/chang9711/BIPS)
* [Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness](https://arxiv.org/abs/2207.10899)<br>:open_mouth:oral:star:[code](https://github.com/pantheon5100/DeACL)
* [RIBAC: Towards Robust and Imperceptible Backdoor Attack against Compact DNN](https://arxiv.org/abs/2208.10608)<br>:star:[code](https://github.com/huyvnphan/ECCV2022-RIBAC)
* [Adversarial Coreset Selection for Efficient Robust Training](https://arxiv.org/abs/2209.05785)
* [Shape Matters: Deformable Patch Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640522.pdf)
* [Enhanced Accuracy and Robustness via Multi-Teacher Adversarial Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640577.pdf)<br>:star:[code](https://github.com/zhaoshiji123/MTARD)
* [GradAuto: Energy-Oriented Attack on Dynamic Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640628.pdf)<br>:star:[code](https://github.com/JianhongPan/GradAuto) 
* [Learning Energy-Based Models with Adversarial Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650204.pdf)
* [Revisiting Outer Optimization in Adversarial Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650240.pdf)
* [One Size Does NOT Fit All: Data-Adaptive Adversarial Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650070.pdf)<br>:star:[code](https://github.com/eccv2022daat/daat)
* [UniCR: Universally Approximated Certified Robustness via Randomized Smoothing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650086.pdf)  
* [ℓ∞-Robustness and Beyond:Unleashing Efficient Adversarial Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710466.pdf)
* [Towards Efficient Adversarial Training on Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730307.pdf)
* [FrequencyLowCut Pooling - Plug & Play against Catastrophic Overfitting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740036.pdf)<br>:star:[code](https://github.com/GeJulia/flc_pooling)
* [TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740053.pdf)<br>:house:[project](https://shivangi-aneja.github.io/projects/tafim)
* 对抗攻击
  * [Frequency Domain Model Augmentation for Adversarial Attack](https://arxiv.org/abs/2207.05382)<br>:star:[code](https://github.com/yuyang-long/SSA)
  * [Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal](https://arxiv.org/abs/2207.08178)<br>:star:[code](https://github.com/thinwayliu/Watermark-Vaccine)
  * [SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness](https://arxiv.org/abs/2207.12391)
  * [Scaling Adversarial Training to Large Perturbation Bounds](https://arxiv.org/abs/2210.09852)<br>:star:[code](https://github.com/val-iisc/OAAT)
  * [Towards Effective and Robust Neural Trojan Defenses via Input Filtering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650277.pdf)
  * [Exploiting the Local Parabolic Landscapes of Adversarial Losses to Accelerate Black-Box Adversarial Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650311.pdf)<br>:star:[code](https://github.com/HoangATran/BABIES)
  * [Robust Network Architecture Search via Feature Distortion Restraining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650120.pdf)
  * [Triangle Attack: A Query-Efficient Decision-Based Adversarial Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650153.pdf)<br>:star:[code](https://github.com/xiaosen-wang/TA)
  * [Adaptive Image Transformations for Transfer-Based Adversarial Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650001.pdf)
* 黑盒
  * [Black-Box Dissector: Towards Erasing-Based Hard-Label Model Stealing Attack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650188.pdf)
  * [An Invisible Black-Box Backdoor Attack through Frequency Domain](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730396.pdf)<br>:star:[code](https://github.com/SoftWiser-group/FTrojan)
* 白盒
  * [Harmonizer: Learning to Perform White-Box Image and Video Harmonization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750688.pdf)
* 对抗样本
  * [Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640714.pdf)<br>:star:[code](https://github.com/ShawnXYang/C-GSP)
 
## Transfer Learning(迁移学习)
* [Factorizing Knowledge in Neural Networks](https://arxiv.org/abs/2207.03337)<br>:star:[code](https://github.com/Adamdad/KnowledgeFactor)
* [SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650137.pdf)<br>:star:[code](https://github.com/AI-secure/SecretGen) 

## Contrastive Learning(对比学习)
* [Network Binarization via Contrastive Learning](https://arxiv.org/abs/2207.02970)<br>:star:[code](https://github.com/42Shawn/CMIM)
* [Adversarial Contrastive Learning via Asymmetric InfoNCE](https://arxiv.org/abs/2207.08374)<br>:star:[code](https://github.com/yqy2001/A-InfoNCE)
* [Fast-MoCo: Boost Momentum-based Contrastive Learning with Combinatorial Patches](https://arxiv.org/abs/2207.08220)<br>:star:[code](https://github.com/orashi/Fast-MoCo)
* [Contrastive Learning for Diverse Disentangled Foreground Generation](https://arxiv.org/abs/2211.02707)
* [Decoupled Contrastive Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860653.pdf)
* [Joint Learning of Localized Representations from Medical Images and Reports](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860670.pdf)
* [Contrasting Quadratic Assignments for Set-Based Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870087.pdf)
* [Generative Subgraph Contrast for Self-Supervised Graph Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900090.pdf)<br>:star:[code](https://github.com/yh-han/GSC) 
  
## Open-set Recognition(开集识别)
* [DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition](https://arxiv.org/abs/2207.02606)
* [Difficulty-Aware Simulator for Open Set Recognition](https://arxiv.org/abs/2207.10024)<br>:star:[code](https://github.com/wjun0830/Difficulty-Aware-Simulator) 

## Machine Learning(机器学习)
* [Predicting is not Understanding: Recognizing and Addressing Underspecification in Machine Learning](https://arxiv.org/abs/2207.02598)

## Feature Learning(联邦学习)
* [SphereFed: Hyperspherical Federated Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860161.pdf)
* [Image Coding for Machines with Omnipotent Feature Learning](https://arxiv.org/abs/2207.01932)
* [Addressing Heterogeneity in Federated Learning via Distributional Transformation](https://arxiv.org/abs/2210.15025)<br>:star:[code](https://github.com/hyhmia/DisTrans)
* [FedLTN: Federated Learning for Sparse and Personalized Lottery Ticket Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720069.pdf)
* [Improving Generalization in Federated Learning by Seeking Flat Minima](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830636.pdf)
* [AdaBest: Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830690.pdf) 
## Meta-Learning(元学习)
* [Bitwidth-Adaptive Quantization-Aware Neural Network Training: A Meta-Learning Approach](https://arxiv.org/abs/2207.10188)
* [Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions](https://arxiv.org/abs/2209.01501)  
* [Learning to Weight Samples for Dynamic Early-exiting Networks](https://arxiv.org/abs/2209.08310)<br>:star:[code](https://github.com/LeapLabTHU/L2W-DEN)  
* [Rethinking Clustering-Based Pseudo-Labeling for Unsupervised Meta-Learning](https://arxiv.org/abs/2209.13635)<br>:star:[code](https://github.com/xingpingdong/PL-CFE)
 
## Model Compression/Knowledge Distillation/Pruning(模型压缩/知识蒸馏/剪枝)
* 知识蒸馏
  * [Knowledge Condensation Distillation](https://arxiv.org/abs/2207.05409)<br>:star:[code](https://github.com/dzy3/KCD)
  * [FedX: Unsupervised Federated Learning with Cross Knowledge Distillation](https://arxiv.org/abs/2207.09158)<br>:star:[code](https://github.com/Sungwon-Han/FEDX)
  * [Black-box Few-shot Knowledge Distillation](https://arxiv.org/abs/2207.12106)<br>:star:[code](https://github.com/nphdang/FS-BBT)
  * [Efficient One Pass Self-distillation with Zipf's Label Smoothing](https://arxiv.org/abs/2207.12980)<br>:star:[code](https://github.com/megvii-research/zipfls)
  * [MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition](https://arxiv.org/abs/2208.05768)<br>:star:[code](https://github.com/winycg/Self-KD-Lib)
  * [Switchable Online Knowledge Distillation](https://arxiv.org/abs/2209.04996)<br>:star:[code](https://github.com/hfutqian/SwitOKD)
  * [Distilling the Undistillable: Learning from a Nasty Teacher](https://arxiv.org/abs/2210.11728)<br>:star:[code](https://github.com/surgan12/NastyAttacks)
  * [Masked Generative Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710053.pdf)<br>:star:[code](https://github.com/yzd-v/MGD)
  * [Prune Your Model before Distill It](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710120.pdf)<br>:star:[code](https://github.com/ososos888/prune-then-distill)
  * [IDa-Det: An Information Discrepancy-Aware Distillation for 1-Bit Detectors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710347.pdf)<br>:star:[code](https://github.com/SteveTsui/IDa-Det)
  * [Deep Ensemble Learning by Diverse Knowledge Distillation for Fine-Grained Object Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710501.pdf)
  * [A Fast Knowledge Distillation Framework for Visual Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840663.pdf)<br>:house:[project](http://zhiqiangshen.com/projects/FKD/index.html)
  * [Self-Regulated Feature Learning via Teacher-Free Feature Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860337.pdf)<br>:house:[project](https://lilujunai.github.io/Teacher-free-Distillation/)
* 量化
  * [Synergistic Self-supervised and Quantization Learning](https://arxiv.org/abs/2207.05432)<br>:open_mouth:oral:star:[code](https://github.com/megvii-research/SSQL-ECCV2022)
  * [PalQuant: Accelerating High-precision Networks on Low-precision Accelerators](https://arxiv.org/abs/2208.01944)<br>:star:[code](https://github.com/huqinghao/PalQuant)
  * [Fine-Grained Data Distribution Alignment for Post-Training Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710070.pdf)<br>:star:[code](https://github.com/zysxmu/FDDA)
  * [Symmetry Regularization and Saturating Nonlinearity for Robust Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710207.pdf)
  * [Mixed-Precision Neural Network Quantization via Learned Layer-Wise Importance](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710260.pdf)
  * [Non-uniform Step Size Quantization for Accurate Post-Training Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710657.pdf)<br>:star:[code](https://github.com/sogh5/SubsetQ)
  * [Towards Accurate Network Quantization with Equivalent Smooth Regularizer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710726.pdf)
  * [Explicit Model Size Control and Relaxation via Smooth Regularization for Mixed-Precision Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720001.pdf)
  * [BASQ: Branch-Wise Activation-Clipping Search Quantization for Sub-4-Bit Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720017.pdf)<br>:star:[code](https://github.com/HanByulKim/BASQ)
  * [RDO-Q: Extremely Fine-Grained Channel-Wise Quantization via Rate-Distortion Optimization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720156.pdf)
  * [PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf)
* 剪枝
  * [FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification](https://arxiv.org/abs/2207.10888)<br>:star:[code](https://github.com/Bernardo1998/FairGRAPE)
  * [Bayesian Optimization with Clustering and Rollback for CNN Auto Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830480.pdf)<br>:star:[code](https://github.com/fanhanwei/BOCR)
  * [Trainability Preserving Neural Structured Pruning](https://arxiv.org/abs/2207.12534)<br>:star:[code](https://github.com/mingsun-tse/TPP)
  * [Interpretations Steered Network Pruning via Amortized Inferred Saliency Maps](https://arxiv.org/abs/2209.02869)<br>:star:[code](https://github.com/Alii-Ganjj/InterpretationsSteeredPruning)
  * [Data-Free Backdoor Removal Based on Channel Lipschitzness](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650171.pdf)<br>:star:[code](https://github.com/rkteddy/channel-Lipschitzness-based-pruning)
  * [Multi-Granularity Pruning for Model Acceleration on Mobile Devices](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710483.pdf)
  * [Ensemble Knowledge Guided Sub-network Search and Fine-Tuning for Filter Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710568.pdf)
  * [Soft Masking for Cost-Constrained Channel Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710640.pdf)<br>:star:[code](https://github.com/NVlabs/SMCP)
  * [Towards Ultra Low Latency Spiking Neural Networks for Vision and Sequential Tasks Using Temporal Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710709.pdf)
  * [CPrune: Compiler-Informed Model Pruning for Efficient Target-Aware DNN Execution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800634.pdf)
  * [Filter Pruning via Feature Discrimination in Deep Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810241.pdf)
* 轻量级
  * [Learning Extremely Lightweight and Robust Model with Differentiable Constraints on Sparsity and Condition Number](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640679.pdf) 
* MC
  * [Patch Similarity Aware Data-Free Quantization for Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710154.pdf)<br>:star:[code](https://github.com/zkkli/PSAQ-ViT)
  * [Disentangled Differentiable Network Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710329.pdf)
  * [Weight Fixing Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710416.pdf)
  * [SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710618.pdf)<br>:star:[code](https://github.com/PeiyanFlying/SPViT)

## Point Cloud(点云)
* [Few 'Zero Level Set'-Shot Learning of Shape Signed Distance Functions in Feature Space](https://arxiv.org/abs/2207.04161)
* [Dynamic 3D Scene Analysis by Point Cloud Accumulation](https://arxiv.org/abs/2207.12394)<br>:star:[code](https://github.com/prs-eth/PCAccumulation):house:[project](https://shengyuh.github.io/eccv22/index.html)
* [Point MixSwap: Attentional Point Cloud Mixing via Swapping Matched Structural Divisions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890587.pdf)<br>:star:[code](https://github.com/ardianumam/PointMixSwap)
* [PointTree: Transformation-Robust Point Cloud Encoder with Relaxed K-D Trees](https://arxiv.org/abs/2208.05962)<br>:star:[code](https://github.com/immortalCO/PointTree)
* [Learning to Generate Realistic LiDAR Point Clouds](https://arxiv.org/abs/2209.03954)<br>:house:[project](https://www.zyrianov.org/lidargen/)
* [PseudoAugment: Learning to Use Unlabeled Data for Data Augmentation in Point Clouds](https://arxiv.org/abs/2210.13428)
* [SPE-Net: Boosting Point Cloud Analysis via Rotation Robustness Enhancement](https://arxiv.org/abs/2211.08250)<br>:star:[code](https://github.com/ZhaofanQiu/SPE-Net)
* [Resolution-Free Point Cloud Sampling Network with Data Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620053.pdf)<br>:star:[code](https://github.com/Tianxinhuang/PCDNet)
* [diffConv: Analyzing Irregular Point Clouds with an Irregular View](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630375.pdf)<br>:star:[code](https://github.com/mmmmimic/diffConvNet)
* [PD-Flow: A Point Cloud Denoising Framework with Normalizing Flows](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630392.pdf)<br>:star:[code](https://github.com/unknownue/pdflow)
* [Shape-Pose Disentanglement Using SE(3)-Equivariant Vector Neurons](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630461.pdf)
* [Revisiting Point Cloud Simplification: A Learnable Feature Preserving Approach](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620573.pdf)
* [Masked Autoencoders for Point Cloud Self-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620591.pdf)<br>:star:[code](https://github.com/Pang-Yatian/Point-MAE)
* [Masked Discrimination for Self-Supervised Learning on Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620645.pdf)<br>:star:[code](https://github.com/haotian-liu/MaskPoint)
* [Meta-Sampler: Almost-Universal yet Task-Oriented Sampling for Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620682.pdf)<br>:star:[code](https://github.com/ttchengab/MetaSampler)
* [Efficient Point Cloud Analysis Using Hilbert Curve](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620717.pdf)
* [RFNet-4D: Joint Object Reconstruction and Flow Estimation from 4D Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830036.pdf)<br>:star:[code](https://github.com/hkust-vgd/RFNet-4D)   
* 3D点云
  * [Autoregressive 3D Shape Generation via Canonical Mapping](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630091.pdf)<br>:star:[code](https://github.com/AnjieCheng/CanonicalVAE)
  * [Point Cloud Domain Adaptation via Masked Local 3D Structure Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630159.pdf)<br>:star:[code](https://github.com/VITA-Group/MLSP)
  * [Exploring the Devil in Graph Spectral Domain for 3D Point Cloud Attacks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630230.pdf)<br>:star:[code](https://github.com/WoodwindHu/GSDA)
  * [Unsupervised Learning of 3D Semantic Keypoints with Mutual Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620521.pdf)<br>:star:[code](https://github.com/YYYYYHC/Learning-Semantic-Keypoints-with-Mutual-Reconstruction)
  * [Few-Shot Class-Incremental Learning for 3D Point Cloud Objects](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800194.pdf)<br>:star:[code](https://github.com/townim-faisal/FSCIL-3D)
  * [Manifold Adversarial Learning for Cross-Domain 3D Shape Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860266.pdf)
* 点云定位
  * [CPO: Change Robust Panorama to Point Cloud Localization](https://arxiv.org/abs/2207.05317)
* 点云分割
 * [Dual Adaptive Transformations for Weakly Supervised Point Cloud Segmentation](https://arxiv.org/abs/2207.09084)
* 点云补全  
  * [SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer](https://arxiv.org/abs/2207.10315)<br>:star:[code](https://github.com/hrzhou2/seedformer)   
  * [FBNet: Feedback Network for Point Cloud Completion](https://arxiv.org/abs/2210.03974)<br>:open_mouth:oral:star:[code](https://github.com/hikvision-research/3DVision/) 
  * [Optimization over Disentangled Encoding: Unsupervised Cross-Domain Point Cloud Completion via Occlusion Factor Manipulation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620504.pdf)<br>:star:[code](https://github.com/azuki-miho/OptDE)
* 点云配准
  * [SuperLine3D: Self-supervised Line Segmentation and Description for LiDAR Point Cloud](https://arxiv.org/abs/2208.01925)<br>:star:[code](https://github.com/zxrzju/SuperLine3D) 
  * [Improving RGB-D Point Cloud Registration by Learning Multi-scale Local Linear Transformation](https://arxiv.org/abs/2208.14893)<br>:star:[code](https://github.com/514DNA/LLT)
  * [PointCLM: A Contrastive Learning-Based Framework for Multi-Instance Point Cloud Registration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690586.pdf)<br>:star:[code](https://github.com/phdymz/PointCLM)
  * [PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700439.pdf)  
* 点云重建
  * [Learning to Train a Point Cloud Reconstruction Network without Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610177.pdf)<br>:star:[code](https://github.com/Tianxinhuang/PCLossNet)
* 点云分类
  * [Improving Adversarial Robustness of 3D Point Cloud Classification Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640663.pdf)<br>:star:[code](https://github.com/GuanlinLee/CCNAMS)
* 点云理解
  * [PointMixer: MLP-Mixer for Point Cloud Understanding](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870611.pdf)<br>:star:[code](https://github.com/LifeBeyondExpectations/ECCV22-PointMixer) 

## SLAM/Augmented Reality/Virtual Reality/Robotics(增强/虚拟现实/机器人)
* 增强现实
  * [LaMAR: Benchmarking Localization and Mapping for Augmented Reality](https://arxiv.org/abs/2210.10770)<br>:star:[code](https://github.com/microsoft/lamar-benchmark):house:[project](https://lamar.ethz.ch/)
* VR
  * [LiP-Flow: Learning Inference-Time Priors for Codec Avatars via Normalizing Flows in Latent Space](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860091.pdf)
  * human volumetric capture(容积捕获)
    * [AvatarCap: Animatable Avatar Conditioned Monocular Human Volumetric Capture](https://arxiv.org/abs/2207.02031)<br>:star:[code](https://github.com/lizhe00/AvatarCap):house:[project](http://www.liuyebin.com/avatarcap/avatarcap.html)
* 虚拟试穿
  * [Single Stage Virtual Try-on via Deformable Attention Flows](https://arxiv.org/abs/2207.09161) 
  * [Dress Code: High-Resolution Multi-Category Virtual Try-On](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680337.pdf)<br>:star:[code](https://github.com/aimagelab/dress-code)
  * [High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770208.pdf)<br>:star:[code](https://github.com/sangyun884/HR-VITON) 
* 视觉定位(相机姿势估计)
  * [MeshLoc: Mesh-Based Visual Localization](https://arxiv.org/abs/2207.10762)<br>:star:[code](https://github.com/tsattler/meshloc_release)
* 机器人
  * [Visual Cross-View Metric Localization with Dense Uncertainty Estimates](https://arxiv.org/abs/2208.08519)<br>:star:[code](https://github.com/tudelft-iv/CrossViewMetricLocalization)


## Optical Flow(光流)
* [Secrets of Event-Based Optical Flow](https://arxiv.org/abs/2207.10022)<br>:star:[code](https://github.com/tub-rip/event_based_optical_flow)
* [Deep 360∘ Optical Flow Estimation Based on Multi-Projection Fusion](https://arxiv.org/abs/2208.00776)
* [Learning Omnidirectional Flow in 360-degree Video via Siamese Representation](https://arxiv.org/abs/2208.03620)<br>:house:[project](https://siamlof.github.io/)
* [Video Interpolation by Event-driven Anisotropic Adjustment of Optical Flow](https://arxiv.org/abs/2208.09127)
* [Learning Omnidirectional Flow in 360° Video via Siamese Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680546.pdf)<br>:house:[project](https://siamlof.github.io/)
* [FlowFormer: A Transformer Architecture for Optical Flow](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770672.pdf)
* [Complementing Brightness Constancy with Deep Networks for Optical Flow Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810120.pdf)
* [Disentangling Architecture and Training for Optical Flow](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820159.pdf)<br>:house:[project](https://autoflow-google.github.io/)
* [A Perturbation-Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820177.pdf)<br>:open_mouth:oral:star:[code](https://github.com/cv-stuttgart/PCFA)
* [Optical Flow Training under Limited Label Budget via Active Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820400.pdf)<br>:star:[code](https://github.com/duke-vision/optical-flow-active-learning-release)
* [S2F2: Single-Stage Flow Forecasting for Future Multiple Trajectories Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820593.pdf)

## Re-identification(重识别)
* 重识别  
  * [Negative Samples are at Large: Leveraging Hard-distance Elastic Loss for Re-identification](https://arxiv.org/abs/2207.09884)
  * [PASS: Part-Aware Self-Supervised Pre-training for Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740192.pdf)<br>:star:[code](https://github.com/CASIA-IVA-Lab/PASS-reID)
  * [Adaptive Cross-Domain Learning for Generalizable Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740209.pdf)<br>:star:[code](https://github.com/peterzpy/ACL-DGReID)
  * [Dynamically Transformed Instance Normalization Network for Generalizable Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740279.pdf)
  * [Mimic Embedding via Adaptive Aggregation: Learning Generalizable Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740362.pdf)<br>:star:[code](https://github.com/xbq1994/META)
  * [Modality Synergy Complement Learning with Cascaded Aggregation for Visible-Infrared Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740450.pdf)<br>:star:[code](https://github.com/bitreidgroup/VI-ReID-MSCLNet)
  * [Cross-Modality Transformer for Visible-Infrared Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740467.pdf)
  * [Optimal Transport for Label-Efficient Visible-Infrared Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840091.pdf) <br>:star:[code](https://github.com/wjm-wjm/OTLA-ReID) 
  * [Counterfactual Intervention Feature Transfer for Visible-Infrared Person Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860371.pdf)
* 行人搜索
  * [OIMNet++: Prototypical Normalization and Localization-aware Learning for Person Search](https://arxiv.org/abs/2207.10320)<br>:house:[project](https://cvlab.yonsei.ac.kr/projects/OIMNetPlus/)
  * [Domain Adaptive Person Search](https://arxiv.org/abs/2207.11898)<br>:open_mouth:oral:star:[code](https://github.com/caposerenity/DAPS)
* 人群计数
  * [An End-to-End Transformer Model for Crowd Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610037.pdf)<br>:house:[project](https://dk-liang.github.io/CLTR/)
  * [Calibration-Free Multi-View Crowd Counting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690224.pdf)
* Visual Search
  * [Target-Absent Human Attention](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640051.pdf)<br>:star:[code](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention)
* 步态识别
  * [MetaGait: Learning to Learn an Omni Sample Adaptive Representation for Gait Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650350.pdf)
  * [GaitEdge: Beyond Plain End-to-End Gait Recognition for Better Practicality](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650368.pdf)<br>:star:[code](https://github.com/ShiqiYu/OpenGait)

## Neural Architecture Search(神经架构搜索)
* [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677)<br>:star:[code](https://github.com/RICE-EIC/SuperTickets)
* [UniNet: Unified Architecture Search with Convolution, Transformer, and MLP](https://arxiv.org/abs/2207.05420)<br>:star:[code](https://github.com/Sense-X/UniNet)
* [ScaleNet: Searching for the Model to Scale](https://arxiv.org/abs/2207.07267)<br>:star:[code](https://github.com/luminolx/ScaleNet)
* [CLOSE: Curriculum Learning On the Sharing Extent Towards Better One-shot NAS](https://arxiv.org/abs/2207.07868)<br>:star:[code](https://github.com/walkerning/aw_nas)
* [Towards Regression-Free Neural Networks for Diverse Compute Platforms](https://arxiv.org/abs/2209.13740)
* [LidarNAS: Unifying and Searching Neural Architectures for 3D Point Clouds](https://arxiv.org/abs/2210.05018)
* [U-Boost NAS: Utilization-Boosted Differentiable Neural Architecture Search](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720172.pdf)<br>:star:[code](https://github.com/yuezuegu/UBoostNAS)
* [A Max-Flow Based Approach for Neural Architecture Search](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800668.pdf)
* [ViTAS: Vision Transformer Architecture Search](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810138.pdf)
* [Learning Where to Look – Generative NAS Is Surprisingly Efficient](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830257.pdf)<br>:star:[code](https://github.com/jovitalukasik/AG-Net)
* [Neural Architecture Search for Spiking Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840036.pdf) 
* [Data-Free Neural Architecture Search via Recursive Label Calibration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840386.pdf)<br>:star:[code](https://github.com/liuzechun/Data-Free-NAS)  
 
## Image Classification(图像分类)
* [Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset](https://arxiv.org/abs/2207.10664)<br>:star:[code](https://github.com/visipedia/ssw60)
* [Centrality and Consistency: Two-Stage Clean Samples Identification for Learning with Instance-Dependent Noisy Labels](https://arxiv.org/abs/2207.14476)<br>:star:[code](https://github.com/uitrbn/TSCSI_IDN)
* [Constructing Balance from Imbalance for Long-tailed Image Recognition](https://arxiv.org/abs/2208.02567)<br>:star:[code](https://github.com/silicx/DLSA)
* [No Token Left Behind: Explainability-Aided Image Classification and Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720329.pdf)<br>:star:[code](https://github.com/apple/ml-no-token-left-behind)
* [Interpretable Image Classification with Differentiable Prototypes Assignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720346.pdf)<br>:star:[code](https://github.com/gmum/ProtoPool)
* [Rotation Regularization without Rotation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850632.pdf)<br>:star:[code](https://github.com/tk1980/StatRot)
* [In Defense of Image Pre-training for Spatiotemporal Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850665.pdf)<br>:star:[code](https://github.com/UCSC-VLAA/Image-Pretraining-for-Video)
* [Augmenting Deep Classifiers with Polynomial Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850682.pdf) 
* [A Dataset Generation Framework for Evaluating Megapixel Image Classifiers & their Explanations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720416.pdf)
* [Cartoon Explanations of Image Classifiers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720439.pdf)
* [Exploring Hierarchical Graph Representation for Large-Scale Zero-Shot Image Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800108.pdf)<br>:house:[project](https://kaiyi.me/p/hgrnet.html)
* [SSBNet: Improving Visual Recognition Efficiency by Adaptive Sampling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810224.pdf)
* [AutoMix: Unveiling the Power of Mixup for Stronger Classifiers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840435.pdf)
* [MaxViT: Multi-axis Vision Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840453.pdf)<br>:star:[code](https://github.com/google-research/maxvit)
* [Self-Feature Distillation with Uncertainty Modeling for Degraded Image Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840544.pdf)
* [Three Things Everyone Should Know about Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840490.pdf)
* [RealPatch: A Statistical Matching Framework for Model Patching with Real Samples](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850144.pdf)<br>:star:[code](https://github.com/wearepal/RealPatch)
* [TDAM: Top-Down Attention Module for Contextually Guided Feature Selection in CNNs](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850255.pdf)<br>:star:[code](https://github.com/shantanuj/TDAM_Top_down_attention_module)
* [Automatic Check-Out via Prototype-Based Classifier Learning from Single-Product Exemplars](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850273.pdf)<br>:star:[code](https://github.com/Hao-Chen-NJUST/PSP)
* 小样本图像分类
  * [Tree Structure-Aware Few-Shot Image Classification via Hierarchical Aggregation](https://arxiv.org/abs/2207.06989)<br>:star:[code](https://github.com/remiMZ/HTS-ECCV22) 
  * [Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519)<br>:star:[code](https://github.com/gaopengcuhk/Tip-Adapter)
  * [Adversarial Feature Augmentation for Cross-domain Few-shot Classification](https://arxiv.org/abs/2208.11021)<br>:star:[code](https://github.com/youthhoo/AFA_For_Few_shot_learning)
  * [Few-Shot Classification with Contrastive Learning](https://arxiv.org/abs/2209.08224)
* 多标签分类
  * [Hyperspherical Learning in Multi-Label Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850038.pdf)<br>:star:[code](https://github.com/TencentYoutuResearch/MultiLabel-HML)
* 长尾分类
  * [SAFA: Sample-Adaptive Feature Augmentation for Long-Tailed Image Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840578.pdf)
  * [Invariant Feature Learning for Generalized Long-Tailed Classification](https://arxiv.org/abs/2207.09504)<br>:star:[code](https://github.com/KaihuaTang/Generalized-Long-Tailed-Benchmarks.pytorch)
  * [Tackling Long-Tailed Category Distribution Under Domain Shifts](https://arxiv.org/abs/2207.10150)<br>:star:[code](https://github.com/guxiao0822/lt-ds):house:[project](https://xiaogu.site/LTDS/)
  * [Identifying Hard Noise in Long-Tailed Sample Distribution](https://arxiv.org/abs/2207.13378)<br>:open_mouth:oral:star:[code](https://github.com/yxymessi/H2E-Framework) 
  * [On Multi-Domain Long-Tailed Recognition, Imbalanced Domain Generalization and Beyond](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800054.pdf)<br>:star:[code](https://github.com/YyzHarry/multi-domain-imbalance)
* 视觉分类
  * [Visual Knowledge Tracing](https://arxiv.org/abs/2207.10157)<br>:star:[code](https://github.com/nkondapa/VisualKnowledgeTracing)
* 细粒度识别
  * [Improving Fine-Grained Visual Recognition in Low Data Regimes via Self-Boosting Attention Mechanism](https://arxiv.org/abs/2208.00617)<br>:star:[code](https://github.com/GANPerf/SAM)  
  * [Zero-Shot Attribute Attacks on Fine-Grained Recognition Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650257.pdf)
  * [Where to Focus: Investigating Hierarchical Attention Relationship for Fine-Grained Visual Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840056.pdf)<br>:star:[code](https://github.com/visiondom/CHRF) 
* 长尾识别
  * [Towards Calibrated Hyper-Sphere Representation via Distribution Overlap Coefficient for Long-tailed Learning](https://arxiv.org/abs/2208.10043)<br>:star:[code](https://github.com/VipaiLab/vMF_OP)
  * [Breadcrumbs: Adversarial Class-Balanced Sampling for Long-Tailed Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840628.pdf)<br>:star:[code](https://github.com/BoLiu-SVCL/Breadcrumbs)
  * [VL-LTR: Learning Class-Wise Visual-Linguistic Representation for Long-Tailed Visual Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850072.pdf)<br>:star:[code](https://github.com/ChangyaoTian/VL-LTR)

 
## Video/Image Super-Resolution(视频/图像超分辨率)
* 跨模态超分辨率
  * [Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution](https://arxiv.org/abs/2207.09156)<br>:star:[code](https://github.com/palmdong/MMSR)
* 图像超分辨率
  * [Image Super-Resolution with Deep Dictionary](https://arxiv.org/abs/2207.09228)<br>:star:[code](https://github.com/shuntama/srdd)
  * [CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution](https://arxiv.org/abs/2207.10345)<br>:star:[code](https://github.com/Cheeun/CADyQ)
  * [Reference-based Image Super-Resolution with Deformable Attention Transformer](https://arxiv.org/abs/2207.11938)<br>:star:[code](https://github.com/caojiezhang/DATSR) 
  * [KXNet: A Model-Driven Deep Neural Network for Blind Super-Resolution](https://arxiv.org/abs/2209.10305)<br>:star:[code](https://github.com/jiahong-fu/KXNet)
  * [Super-Resolution by Predicting Offsets: An Ultra-Efficient Super-Resolution Network for Rasterized Images](https://arxiv.org/abs/2210.04198)  
  * [Boosting Event Stream Super-Resolution with a Recurrent Neural Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660461.pdf) 
  * [Learning Series-Parallel Lookup Tables for Efficient Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770309.pdf)<br>:star:[code](https://github.com/zhjy2016/SPLUT)
  * [Efficient Long-Range Attention Network for Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770653.pdf)<br>:star:[code](https://github.com/xindongzhang/ELAN)
  * [Metric Learning Based Interactive Modulation for Real-World Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770727.pdf)<br>:star:[code](https://github.com/TencentARC/MM-RealSR)
  * [Dynamic Dual Trainable Bounds for Ultra-Low Precision Super-Resolution Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780001.pdf)<br>:star:[code](https://github.com/zysxmu/DDTB)
  * [Perception-Distortion Balanced ADMM Optimization for Single-Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780106.pdf)<br>:star:[code](https://github.com/Yuehan717/PDASR)
  * [Uncertainty Learning in Kernel Estimation for Multi-stage Blind Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780141.pdf)
  * [MuLUT: Cooperating Multiple Look-Up Tables for Efficient Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780234.pdf)
  * [Adaptive Patch Exiting for Scalable Single Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf)<br>:star:[code](https://github.com/littlepure2333/APE)
  * [From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780368.pdf)<br>:star:[code](https://github.com/csxmli2016/ReDegNet)
  * [Unfolded Deep Kernel Estimation for Blind Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780493.pdf)<br>:star:[code](https://github.com/natezhenghy/UDKE)
  * [Efficient and Degradation-Adaptive Network for Real-World Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780563.pdf)<br>:star:[code](https://github.com/csjliang/DASR)
  * [Self-Supervised Learning for Real-World Super-Resolution from Dual Zoomed Observations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780599.pdf)<br>:star:[code](https://github.com/cszhilu1998/SelfDZSR)
  * [Restore Globally, Refine Locally: A Mask-Guided Scheme to Accelerate Super-Resolution Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790072.pdf)<br>:star:[code](https://github.com/huxiaotaostasy/MGA-scheme)
  * [Compiler-Aware Neural Architecture Search for On-Mobile Real-Time Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790089.pdf)<br>:star:[code](https://github.com/wuyushuwys/compiler-aware-nas-sr)
  * [KXNet: A Model-Driven Deep Neural Network for Blind Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790230.pdf)<br>:star:[code](https://github.com/jiahong-fu/KXNet)
  * [ARM: Any-Time Super-Resolution Method](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790248.pdf)<br>:star:[code](https://github.com/chenbong/ARM-Net)
  * [D2C-SR: A Divergence to Convergence Approach for Real-World Image Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790370.pdf)<br>:star:[code](https://github.com/megvii-research/D2C-SR)
  * [RRSR:Reciprocal Reference-Based Image Super-Resolution with Progressive Feature Alignment and Selection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790637.pdf)
* 视频超分辨率
  * [Towards Interpretable Video Super-Resolution via Alternating Optimization](https://arxiv.org/abs/2207.10765)<br>:star:[code](https://github.com/caojiezhang/DAVSR)
  * [Learning Spatiotemporal Frequency-Transformer for Compressed Video Super-Resolution](https://arxiv.org/abs/2208.03012)<br>:star:[code](https://github.com/researchmm/FTVSR)
  * [Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset](https://arxiv.org/abs/2209.12475)<br>:star:[code](https://github.com/zmzhang1998/Real-RawVSR)
  * [A Codec Information Assisted Framework for Efficient Compressed Video Super-Resolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770224.pdf) 

## 25.Autonomous vehicles(自动驾驶)
* 车辆轨迹预测
  * [Hierarchical Latent Structure for Multi-Modal Vehicle Trajectory Forecasting](https://arxiv.org/abs/2207.04624)<br>:star:[code](https://github.com/d1024choi/HLSTrajForecast)
  * [Action-based Contrastive Learning for Trajectory Prediction](https://arxiv.org/abs/2207.08664)
  * [D2-TPred: Discontinuous Dependency for Trajectory Prediction under Traffic Lights](https://arxiv.org/abs/2207.10398)<br>:star:[code](https://github.com/VTP-TL/D2-TPred)
  * [AdvDO: Realistic Adversarial Attacks for Trajectory Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650036.pdf)<br>:house:[project](https://robustav.github.io/RobustPred/)
* 自动驾驶
  * [ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning](https://arxiv.org/abs/2207.07601)<br>:star:[code](https://github.com/OpenPerceptionX/ST-P3)
  * [Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction](https://arxiv.org/abs/2207.09705)
  * [Dfferentiable Raycasting for Self-supervised Occupancy Forecasting](https://arxiv.org/abs/2210.01917)<br>:star:[code](https://github.com/tarashakhurana/emergent-occ-forecasting)
  * [Self-Distillation for Robust LiDAR Semantic Segmentation in Autonomous Driving](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880650.pdf)<br>:star:[code](https://github.com/jialeli1/lidarseg3d)
  * [Motion Inspired Unsupervised Perception and Prediction in Autonomous Driving](https://arxiv.org/abs/2210.08061)
  * [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690001.pdf)<br>:star:[code](https://github.com/zhiqi-li/BEVFormer)
  * [Point Cloud Compression with Range Image-Based Entropy Model for Autonomous Driving](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820315.pdf)
* 轨迹预测
  * [Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction](https://arxiv.org/abs/2207.09953)<br>:star:[code](https://github.com/inhwanbae/GPGraph)
  * [Aware of the History: Trajectory Forecasting with the Local Behavior Data](https://arxiv.org/abs/2207.09646)<br>:star:[code](https://github.com/Kay1794/LocalBehavior-based-trajectory-prediction)
  * [Social-SSL: Self-Supervised Cross-Sequence Representation Learning Based on Transformers for Multi-agent Trajectory Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820227.pdf)<br>:star:[code](https://github.com/Sigta678/Social-SSL)
  * [Sequential Multi-View Fusion Network for Fast LiDAR Point Motion Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820282.pdf)
  * [Social-Implicit: Rethinking Trajectory Prediction Evaluation and the Effectiveness of Implicit Maximum Likelihood Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820451.pdf)<br>:star:[code](https://github.com/abduallahmohamed/Social-Implicit/)
  * [View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820661.pdf)<br>:star:[code](https://github.com/cocoon2wong/Vertical) 
* 车道线检测
  * [RCLane: Relay Chain Prediction for Lane Detection](https://arxiv.org/abs/2207.09399)
* 行人轨迹预测
  * [Human Trajectory Prediction via Neural Social Physics](https://arxiv.org/abs/2207.10435)<br>:star:[code](https://github.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics)
  * [SocialVAE: Human Trajectory Prediction Using Timewise Latents](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640504.pdf)<br>:star:[code](https://github.com/xupei0610/SocialVAE):house:[project](https://motion-lab.github.io/SocialVAE/)
* 车辆重识别
  * [Unstructured Feature Decoupling for Vehicle Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740328.pdf)<br>:star:[code](https://github.com/damo-cv/UFDN-Reid)  

<a name="24"/>

## 24.UAV/Remote Sensing/Satellite Image(无人机/遥感/卫星图像)
* 遥感
  * [Tomography of Turbulence Strength Based on Scintillation Imaging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670464.pdf) 
  * [TD-Road: Top-Down Road Network Extraction with Holistic Graph Construction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690553.pdf)
  
<a name="23"/>

## 23.Medical Image(医学影像)
* [The Surprisingly Straightforward Scene Text Removal Method With Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis](https://arxiv.org/abs/2210.07489)<br>:star:[code](https://github.com/wyndwarrior/autoregressive-bbox):house:[project](https://bbox.yuxuanliu.com/)
* 医学图像分割
  * [Personalizing Federated Medical Image Segmentation via Local Calibration](https://arxiv.org/abs/2207.04655)<br>:star:[code](https://github.com/jcwang123/FedLC)
  * [Learning Topological Interactions for Multi-Class Medical Image Segmentation](https://arxiv.org/abs/2207.09654)<br>:open_mouth:oral:star:[code](https://github.com/TopoXLab/TopoInteraction)
  * [Generalizable Medical Image Segmentation via Random Amplitude Mixup and Domain-Specific Image Restoration](https://arxiv.org/abs/2208.03901)<br>:star:[code](https://github.com/zzzqzhou/RAM-DSIR)
  * [PointScatter: Point Set Representation for Tubular Structure Extraction](https://arxiv.org/abs/2209.05774)<br>:open_mouth:oral:star:[code](https://github.com/zhangzhao2022/pointscatter)
  * [Dual Contrastive Learning with Anatomical Auxiliary Supervision for Few-Shot Medical Image Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800406.pdf)<br>:star:[code](https://github.com/cvszusparkle/AAS-DCL_FS)
  * [Auto-FedRL: Federated Hyperparameter Optimization for Multi-Institutional Medical Image Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810431.pdf)<br>:star:[code](https://github.com/nvidia/nvflare/research/auto-fedrl)
  * [Med-DANet: Dynamic Architecture Network for Efficient Medical Volumetric Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810499.pdf)<br>:star:[code](https://github.com/Wenxuan-1119/Med-DANet)
  * [CXR Segmentation by AdaIN-Based Domain Adaptation and Knowledge Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810619.pdf)
* 放射科报告生成
  * [Cross-modal Prototype Driven Network for Radiology Report Generation](https://arxiv.org/abs/2207.04818)<br>:star:[code](https://github.com/Markin-Wang/XProNet)
* 密集预测
  * [ConCL: Concept Contrastive Learning for Dense Prediction Pre-training in Pathology Images](https://arxiv.org/abs/2207.06733)<br>:star:[code](https://github.com/TencentAILabHealthcare/ConCL)
* retinal image matching(视网膜图像匹配)
  * [Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching](https://arxiv.org/abs/2207.07932)<br>:star:[code](https://github.com/ruc-aimc-lab/SuperRetina)
* 支架追踪
  * [Robust Landmark-based Stent Tracking in X-ray Fluoroscopy](https://arxiv.org/abs/2207.09933)
* 病变检测
  * [Check and Link: Pairwise Lesion Correspondence Guides Mammogram Mass Detection](https://arxiv.org/abs/2209.05809)
* 医学图像分析
  * [UniMiSS: Universal Medical Self-Supervised Learning via Breaking Dimensionality Barrier](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810551.pdf)<br>:star:[code](https://github.com/YtongXie/UniMiSS-code)
  * [K-SALSA: K-Anonymous Synthetic Averaging of Retinal Images via Local Style Alignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810652.pdf)<br>:star:[code](https://github.com/hcholab/k-salsa)
* 医学图像分类
  * [Differentiable Zooming for Multiple Instance Learning on Whole-Slide Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810689.pdf)<br>:star:[code](https://github.com/histocartography/zoommil)
  * 疾病分类
    * [RadioTransformer: A Cascaded Global-Focal Transformer for Visual Attention-Guided Disease Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810669.pdf)<br>:star:[code](https://github.com/bmi-imaginelab/radiotransformer)
* 医学关键点定位
  * [One-Shot Medical Landmark Localization by Edge-Guided Transform and Noisy Landmark Refinement](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810466.pdf)<br>:star:[code](https://github.com/GoldExcalibur/EdgeTrans4Mark)


<a name="22"/>

## 22.OCR
* [Levenshtein OCR](https://arxiv.org/abs/2209.03594)
* 文本识别
  * [Text-DIAE: Degradation Invariant Autoencoders for Text Recognition and Document Enhancement](https://arxiv.org/abs/2203.04814)
* 手写数学表达式识别
  * [CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2207.04410)<br>:star:[code](https://github.com/Green-Wood/CoMER)
  * [When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2207.11463)<br>:star:[code](https://github.com/LBH1024/CAN)
* 场景文本检测
  * [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966)<br>:star:[code](https://github.com/baudm/parseq)
  * [Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting](https://arxiv.org/abs/2207.06694)<br>:star:[code](https://github.com/hikopensource/DAVAR-Lab-OCR/)
  * [SGBANet: Semantic GAN and Balanced Attention Network for Arbitrarily Oriented Scene Text Recognition](https://arxiv.org/abs/2207.10256)
  * [Optimal Boxes: Boosting End-to-End Scene Text Recognition by Adjusting Annotated Bounding Boxes via Reinforcement Learning](https://arxiv.org/abs/2207.11934)
  * [Contextual Text Block Detection towards Scene Text Understanding](https://arxiv.org/abs/2207.12955)<br>:house:[project](https://sg-vilab.github.io/publication/xue2022contextual/)  
  * [Toward Understanding WordArt: Corner-Guided Transformer for Scene Text Recognition](https://arxiv.org/abs/2208.00438)<br>:open_mouth:oral:star:[code](https://github.com/xdxie/WordArt)
  * [GLASS: Global to Local Attention for Scene-Text Spotting](https://arxiv.org/abs/2208.03364)<br>:star:[code](https://github.com/amazon-research/glass-text-spotting)
  * [Multi-Granularity Prediction for Scene Text Recognition](https://arxiv.org/abs/2209.03592)
  * [Pure Transformer with Integrated Experts for Scene Text Recognition](https://arxiv.org/abs/2211.04963)
  * [Background-Insensitive Scene Text Recognition with Text Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850161.pdf)
  * [Detecting Tampered Scene Text in the Wild](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880214.pdf)<br>:star:[code](https://github.com/wangyuxin87/Tampered-IC13)
  * [Language Matters: A Weakly Supervised Vision-Language Pre-training Approach for Scene Text Detection and Spotting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880282.pdf)
  * [TextAdaIN: Paying Attention to Shortcut Learning in Text Recognizers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880423.pdf)
  * [Multi-modal Text Recognition Networks: Interactive Enhancements between Visual and Semantic Features](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880442.pdf)<br>:star:[code](https://github.com/byeonghu-na/MATRN)
  * [OCR-Free Document Understanding Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880493.pdf)<br>:star:[code](https://github.com/clovaai/donut)
* 视频文本检测
  * [Real-time End-to-End Video Text Spotter with Contrastive Representation Learning](https://arxiv.org/abs/2207.08417)<br>:star:[code](https://github.com/weijiawu/CoText)
* 文本检测
  * [Unitail: Detecting, Reading, and Matching in Retail Scene"](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670695.pdf)<br>:house:[project](https://unitedretail.github.io/)
* 文件图像矫正
  * [Geometric Representation Learning for Document Image Rectification](https://arxiv.org/abs/2210.08161)<br>:star:[code](https://github.com/fh2019ustc/DocGeoNet) 

<a name="21"/>

## 21.Semi/self-supervised learning(半/自监督)
* 无监督
  * [Contrastive Positive Mining for Unsupervised 3D Action Representation Learning](https://arxiv.org/abs/2208.03497)
  * [Dense Siamese Network for Dense Unsupervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900460.pdf)<br>:star:[code](https://github.com/ZwwWayne/DenseSiam)
  * [Contrastive Positive Mining for Unsupervised 3D Action Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640035.pdf)
  * [Relative Contrastive Loss for Unsupervised Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870001.pdf)
* 弱监督
  * [Acknowledging the Unknown for Multi-Label Learning with Single Positive Labels](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840418.pdf)<br>:star:[code](https://github.com/Correr-Zhou/SPML-AckTheUnknown)
* 自监督
  * [GOCA: Guided Online Cluster Assignment for Self-Supervised Video Representation Learning](https://arxiv.org/abs/2207.10158)<br>:star:[code](https://github.com/Seleucia/goca)
  * [Mc-BEiT: Multi-Choice Discretization for Image BERT Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900229.pdf)<br>:star:[code](https://github.com/lixiaotong97/mc-BEiT)
  * [What to Hide from Your Students: Attention-Guided Masked Image Modeling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900299.pdf)<br>:star:[code](https://github.com/gkakogeorgiou/attmask)
  * [Constrained Mean Shift Using Distant Yet Related Neighbors for Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910021.pdf)<br>:star:[code](https://github.com/UCDvision/CMSF)
  * [Semantic-Aware Fine-Grained Correspondence](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910093.pdf)
  * [Self-Supervised Classification Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910112.pdf)<br>:star:[code](https://github.com/elad-amrani/self-classifier)
  * [Dual-Domain Self-Supervised Learning and Model Adaption for Deep Compressive Imaging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900406.pdf) 
  * [SdAE: Self-distillated Masked Autoencoder](https://arxiv.org/abs/2208.00449)<br>:star:[code](https://github.com/AbrahamYabo/SdAE)
  * [RDA: Reciprocal Distribution Alignment for Robust SSL](https://arxiv.org/abs/2208.04619)<br>:star:[code](https://github.com/NJUyued/RDA4RobustSSL)
  * [Motion Sensitive Contrastive Learning for Self-supervised Video Representation](https://arxiv.org/abs/2208.06105)
  * [Towards Efficient and Effective Self-Supervised Learning of Visual Representations](https://arxiv.org/abs/2210.09866)<br>:star:[code](https://github.com/val-iisc/EffSSL)
  * [Unifying Visual Contrastive Learning for Object Recognition from a Graph Perspective](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860635.pdf)
  * [The Challenges of Continuous Self-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860687.pdf)
  * [GeoRefine: Self-Supervised Online Depth Refinement for Accurate Dense Mapping](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610354.pdf)
  * [Fusion from Decomposition: A Self-Supervised Decomposition Approach for Image Fusion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780706.pdf)
  * [DNA: Improving Few-Shot Transfer Learning with Low-Rank Decomposition and Alignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800229.pdf)<br>:star:[code](https://github.com/VITA-Group/DnA)
  * [Self-Supervised Learning of Visual Graph Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830359.pdf)<br>:star:[code](https://github.com/Thinklab-SJTU/ThinkMatch-SCGM)
  * [DisCo: Remedying Self-Supervised Learning on Lightweight Models with Distilled Contrastive Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860233.pdf)<br>:star:[code](https://github.com/Yuting-Gao/DisCo-pytorch)
  * [SLIP: Self-Supervision Meets Language-Image Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860514.pdf)<br>:star:[code](https://github.com/facebookresearch/SLIP)
  * [Discovering Deformable Keypoint Pyramids](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860531.pdf)<br>:star:[code](https://github.com/jianingq/KeypointPyramids)
* 半监督
  * [Towards Realistic Semi-Supervised Learning](https://arxiv.org/abs/2207.02269)
  * [OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning](https://arxiv.org/abs/2207.02261)
  * [Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning](https://arxiv.org/abs/2207.12535)<br>:star:[code](https://github.com/xinleihe/Semi-Leak)
  * [ConMatch: Semi-Supervised Learning with Confidence-Guided Consistency Regularization](https://arxiv.org/abs/2208.08631)<br>:star:[code](https://github.com/JiwonCocoder/ConMatch)
  * [Vibration-Based Uncertainty Estimation for Learning from Limited Supervision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900160.pdf)
  * [Unsupervised Selective Labeling for More Effective Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900423.pdf)
  * [RDA: Reciprocal Distribution Alignment for Robust Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900527.pdf)<br>:star:[code](https://github.com/NJUyued/RDA4RobustSSL)
  * [Semi-Supervised Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900596.pdf)<br>:star:[code](https://github.com/wengzejia1/Semiformer)
  * [CA-SSL: Class-Agnostic Semi-Supervised Learning for Detection and Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910057.pdf)
  * [RVSL: Robust Vehicle Similarity Learning in Real Hazy Scenes Based on Semi-supervised Learning](https://arxiv.org/abs/2209.08630)<br>:star:[code](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22)
  * [Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810586.pdf)
* 监督学习
  * [Supervised Attribute Information Removal and Reconstruction for Image Manipulation](https://arxiv.org/abs/2207.06555)<br>:star:[code](https://github.com/NannanLi999/AIRR)
  * [Tailoring Self-Supervision for Supervised Learning](https://arxiv.org/abs/2207.10023)<br>:star:[code](https://github.com/wjun0830/Localizable-Rotation)  

<a name="20"/>

## 20.Face(人脸)
* [Effective Presentation Attack Detection Driven by Face Related Task](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650400.pdf)<br>:star:[code](https://github.com/WentianZhang-ML/FRT-PAD)
* [Facial Depth and Normal Estimation Using Single Dual-Pixel Camera](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680176.pdf)<br>:star:[code](https://github.com/MinJunKang/DualPixelFace)
* [StyleFace: Towards Identity-Disentangled Face Generation on Megapixels](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760281.pdf)
* [Augmentation of rPPG Benchmark Datasets: Learning to Remove and Embed rPPG Signals via Double Cycle Consistent Learning from Unpaired Facial Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760351.pdf)<br>:star:[code](https://github.com/nthumplab/RErPPGNet)
* [Custom Structure Preservation in Face Aging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760541.pdf)
* deepfake检测
  * [Detecting and Recovering Sequential DeepFake Manipulation](https://arxiv.org/abs/2207.02204)<br>:star:[code](https://github.com/rshaojimmy/SeqDeepFake):house:[project](https://rshaojimmy.github.io/Projects/SeqDeepFake)
  * [Explaining Deepfake Detection by Analysing Image Matching](https://arxiv.org/abs/2207.09679)<br>:star:[code](https://github.com/megvii-research/FST-Matching) 
  * [Hierarchical Contrastive Inconsistency Learning for Deepfake Video Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720588.pdf)
* 三维人脸
  * [Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation](https://arxiv.org/abs/2207.09019)<br>:star:[code](https://github.com/gerwang/facial-detail-manipulation) 
* 活体检测
  * [Generative Domain Adaptation for Face Anti-Spoofing](https://arxiv.org/abs/2207.10015)
  * [Multi-domain Learning for Updating Face Anti-spoofing Models](https://arxiv.org/abs/2208.11148)<br>:star:[code](https://github.com/CHELSEA234/Multi-domain-learning-FAS)
  * [Source-Free Domain Adaptation with Contrastive Domain Alignment and Self-Supervised Exploration for Face Anti-Spoofing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720506.pdf)<br>:star:[code](https://github.com/YuchenLiu98/ECCV2022-SDA-FAS)
  * [Adaptive Transformers for Robust Few-Shot Cross-Domain Face Anti-Spoofing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730037.pdf)
* 人脸识别
  * [Controllable and Guided Face Synthesis for Unconstrained Face Recognition](https://arxiv.org/abs/2207.10180)<br>:star:[code](https://github.com/liuf1990/CFSM):house:[project](http://cvlab.cse.msu.edu/project-cfsm.html)
  * [Towards Robust Face Recognition with Comprehensive Search](https://arxiv.org/abs/2208.13600)
  * [BoundaryFace: A mining framework with noise label self-correction for Face Recognition](https://arxiv.org/abs/2210.04567)<br>:star:[code](https://github.com/SWJTU-3DVision/BoundaryFace)
  * [Privacy-Preserving Face Recognition with Learnable Privacy Budgets in Frequency Domain](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720471.pdf)<br>:star:[code](https://github.com/Tencent/TFace/tree/master/recognition/tasks/dctdp)
  * [OneFace: One Threshold for All](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720539.pdf)
  * [AgeTransGAN for Facial Age Transformation with Rectified Performance Metrics](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720573.pdf)<br>:star:[code](https://github.com/AvLab-CV/AgeTransGAN)
  * [Teaching Where to Look: Attention Similarity Knowledge Distillation for Low Resolution Face Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720622.pdf)<br>:star:[code](https://github.com/gist-ailab/teaching-where-to-look)
  * [CoupleFace: Relation Matters for Face Recognition Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720674.pdf)
  * [Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730072.pdf)<br>:house:[project](https://trust.is.tue.mpg.de/)
  * [Pre-training Strategies and Datasets for Facial Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730109.pdf)
  * [Unsupervised and Semi-Supervised Bias Benchmarking in Face Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730288.pdf)
* 人脸聚类
  * [On Mitigating Hard Clusters for Face Clustering](https://arxiv.org/abs/2207.11895)<br>:open_mouth:oral:star:[code](https://github.com/echoanran/On-Mitigating-Hard-Clusters)
* 说话人脸合成
  * [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770086.pdf)<br>:star:[code](https://github.com/FeiiYin/StyleHEAT)
* 谈话头像合成
  * [Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis](https://arxiv.org/abs/2207.11770)<br>:star:[code](https://github.com/sstzal/DFRF):house:[project](https://sstzal.github.io/DFRF/)
* 人脸姿势估计
  * [Towards Unbiased Label Distribution Learning for Facial Pose Estimation Using Anisotropic Spherical Gaussian](https://arxiv.org/abs/2208.09122)
* 人脸交换
  * [StyleSwap: Style-Based Generator Empowers Robust Face Swapping](https://arxiv.org/abs/2209.13514)<br>:star:[code](https://github.com/Seanseattle/StyleSwap):house:[project](https://hangz-nju-cuhk.github.io/projects/StyleSwap)
  * [Designing One Unified Framework for High-Fidelity Face Reenactment and Swapping](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750053.pdf)<br>:star:[code](https://github.com/xc-csc101/UniFace)
* 假脸检测
  * [UIA-ViT: Unsupervised Inconsistency-Aware Method based on Vision Transformer for Face Forgery Detection](https://arxiv.org/abs/2210.12752)<br>:open_mouth:oral
  * [An Information Theoretic Approach for Attention-Driven Face Forgery Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740105.pdf)
  * [Exploring Disentangled Content Information for Face Forgery Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740122.pdf)
* 人脸捕捉
  * [Practical and Scalable Desktop-Based High-Quality Facial Capture](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660512.pdf)
* 人脸表情识别
  * [How to Synthesize a Large-Scale and Trainable Micro-Expression Dataset?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680037.pdf)<br>:star:[code](https://github.com/liuyvchi/MiE-X)
  * [Teaching with Soft Label Smoothing for Mitigating Noisy Labels in Facial Expressions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720639.pdf)<br>:star:[code](https://github.com/toharl/soft)
  * [Order Learning Using Partially Ordered Data via Chainization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730199.pdf)<br>:star:[code](https://github.com/seon92/Chainization)
  * [Emotion-Aware Multi-View Contrastive Learning for Facial Emotion Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730181.pdf)<br>:star:[code](https://github.com/kdhht2334/AVCE_FER)
  * [Learn-to-Decompose: Cascaded Decomposition Network for Cross-Domain Few-Shot Facial Expression Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790672.pdf)<br>:star:[code](https://github.com/zouxinyi0625/CDNet)
  * [Learn from All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860406.pdf)<br>:star:[code](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency)
* 三维人脸重建
  * [REALY: Rethinking the Evaluation of 3D Face Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680072.pdf)<br>:star:[code](https://github.com/czh-98/REALY):house:[project](https://www.realy3dface.com/)
  * [AU-Aware 3D Face Reconstruction through Personalized AU-Specific Blendshape Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730001.pdf)
  * [3D Face Reconstruction with Dense Landmarks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730162.pdf)
  * [Towards Metrical Reconstruction of Human Faces](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730249.pdf)<br>:house:[project](https://zielon.github.io/mica/)
* 人脸重现
  * [Face2Faceρ: Real-Time High-Resolution One-Shot Face Reenactment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730055.pdf)
* 人脸身份操作
  * [MFIM: Megapixel Facial Identity Manipulation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730145.pdf)
* 人脸纹理合成与重建
  * [Unsupervised High-Fidelity Facial Texture Generation and Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730215.pdf)<br>:star:[code](https://github.com/ronslos/Unsupervised-High-Fidelity-Facial-Texture-Generation-and-Reconstruction)
* 人脸恢复
  * [VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780124.pdf)<br>:star:[code](https://github.com/TencentARC/VQFR/)

<a name="19"/>

## 19.Image Synthesis/Generation(图像合成)
* [Injecting 3D Perception of Controllable NeRF-GAN into StyleGAN for Editable Portrait Image Synthesis](https://arxiv.org/abs/2207.10257)<br>:star:[code](https://github.com/jgkwak95/SURF-GAN):house:[project](https://jgkwak95.github.io/surfgan/)
* [GALA: Toward Geometry-and-Lighting-Aware Object Search for Compositing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870665.pdf)
* [Auto-regressive Image Synthesis with Integrated Quantization](https://arxiv.org/abs/2207.10776)<br>:open_mouth:oral
* [Paint2Pix: Interactive Painting based Progressive Image Synthesis and Editing](https://arxiv.org/abs/2208.08092)<br>:star:[code](https://github.com/1jsingh/paint2pix)
* [Improved Masked Image Generation with Token-Critic](https://arxiv.org/abs/2209.04439)
* [Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation](https://arxiv.org/abs/2209.05968)
* [SCAM! Transferring humans between images with Semantic Cross Attention Modulation](https://arxiv.org/abs/2210.04883)<br>:house:[project](https://imagine.enpc.fr/~dufourn/publications/scam.html)
* [PixelFolder: An Efficient Progressive Pixel Synthesis Network for Image Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740626.pdf)<br>:star:[code](https://github.com/BlingHe/PixelFolder)
* [Adaptive Feature Interpolation for Low-Shot Image Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750251.pdf)
* [Few-Shot Image Generation with Mixup-Based Distance Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750561.pdf)<br>:star:[code](https://github.com/reyllama/mixdl)
* [Multimodal Conditional Image Synthesis with Product-of-Experts GANs](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760089.pdf)<br>:house:[project](https://deepimagination.github.io/PoE-GAN/)
* [Any-Resolution Training for High-Resolution Image Synthesis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760158.pdf)<br>:house:[project](https://chail.github.io/anyres-gan/)
* [3D-Aware Indoor Scene Synthesis with Depth Priors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760385.pdf)<br>:house:[project](https://vivianszf.github.io/depthgan)
* 图像生成
  * [DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta](https://arxiv.org/abs/2207.10271)<br>:star:[code](https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation)
  * [Scraping Textures from Natural Images for Synthesis and Editing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750389.pdf)<br>:house:[project](https://sunshineatnoon.github.io/texture-from-image/)
  * [CoGS: Controllable Generation and Search from Sketch and Style](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760610.pdf) 
  * [Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770600.pdf)
  * [Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830171.pdf)<br>:star:[code](https://github.com/samb-t/unleashing-transformers)
* 样本引导下的图像生成
  * [DynaST: Dynamic Sparse Transformer for Exemplar-Guided Image Generation](https://arxiv.org/abs/2207.06124)<br>:star:[code](https://github.com/Huage001/DynaST)
* 文本-图像合成
  * [StoryDALL-E: Adapting Pretrained Text-to-Image Transformers for Story Continuation](https://arxiv.org/abs/2209.06192)<br>:star:[code](https://github.com/adymaharana/storydalle)
  * [Make-a-Scene: Scene-Based Text-to-Image Generation with Human Priors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750087.pdf)<br>:tv:[video](https://www.youtube.com/watch?v=N4BagnXzPXY)
* 从文本描述中生成不同的人类动作
  * [TEMOS: Generating Diverse Human Motions from Textual Descriptions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820468.pdf)<br>:open_mouth:oral:star:[code](https://github.com/Mathux/TEMOS):house:[project](https://mathis.petrovich.fr/temos/)

<a name="18"/>

## 18.Image-to-Image Translation(图像到图像翻译)
* [VecGAN: Image-to-Image Translation with Interpretable Latent Directions](https://arxiv.org/abs/2207.03411)
* [Vector Quantized Image-to-Image Translation](https://arxiv.org/abs/2207.13286)<br>:star:[code](https://github.com/cyj407/VQ-I2I):house:[project](https://cyj407.github.io/VQ-I2I/)
* [Ultra-high-resolution unpaired stain transformation via Kernelized Instance Normalization](https://arxiv.org/abs/2208.10730)<br>:star:[code](https://github.com/Kaminyou/URUST)
* [Unpaired Image Translation via Vector Symbolic Architectures](https://arxiv.org/abs/2209.02686)<br>:open_mouth:oral:star:[code](https://github.com/facebookresearch/vsait) 
* [Bi-Level Feature Alignment for Versatile Image Translation and Manipulation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760210.pdf)
* [ManiFest: Manifold Deformation for Few-Shot Image Translation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770443.pdf)<br>:star:[code](https://github.com/astra-vision/ManiFest)
* 图像翻译
  * [BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks](https://arxiv.org/abs/2207.06873)<br>:star:[code](https://github.com/ExplainableML/BayesCap)
  * [Multi-Curve Translator for High-Resolution Photorealistic Image Translation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750124.pdf)

<a name="17"/>

## 17.GAN
* [RepMix: Representation Mixing for Robust Attribution of Synthesized Images](https://arxiv.org/abs/2207.02063)<br>:star:[code](https://github.com/TuBui/image_attribution)
* [FakeCLR: Exploring Contrastive Learning for Solving Latent Discontinuity in Data-Efficient GANs](https://arxiv.org/abs/2207.08630)<br>:star:[code](https://github.com/iceli1007/FakeCLR)
* [Generative Multiplane Images: Making a 2D GAN 3D-Aware](https://arxiv.org/abs/2207.10642)<br>:star:[code](https://github.com/apple/ml-gmpi):house:[project](https://xiaoming-zhao.github.io/projects/gmpi/)
* [Generator Knows What Discriminator Should Learn in Unconditional GANs](https://arxiv.org/abs/2207.13320)<br>:star:[code](https://github.com/naver-ai/GGDR)
* [Hierarchical Semantic Regularization of Latent Spaces in StyleGANs](https://arxiv.org/abs/2208.03764)<br>:star:[code](https://drive.google.com/file/d/1gzHTYTgGBUlDWyN_Z3ORofisQrHChg_n/view):house:[project](https://sites.google.com/view/hsr-eccv22)
* [Mind the Gap in Distilling StyleGANs](https://arxiv.org/abs/2208.08840)<br>:star:[code](https://github.com/xuguodong03/StyleKD)
* [FurryGAN: High Quality Foreground-aware Image Synthesis](https://arxiv.org/abs/2208.10422)<br>:house:[project](https://jeongminb.github.io/FurryGAN/)
* [Improving GANs for Long-Tailed Data through Group Spectral Regularization](https://arxiv.org/abs/2208.09932)<br>:star:[code](https://drive.google.com/file/d/1aG48i04Q8mOmD968PAgwEvPsw1zcS4Gk/view):house:[project](https://sites.google.com/view/gsr-eccv22)
* [3D-FM GAN: Towards 3D-Controllable Face Manipulation](https://arxiv.org/abs/2208.11257)<br>:house:[project](https://lychenyoko.github.io/3D-FM-GAN-Webpage/)
* [Exploring Gradient-based Multi-directional Controls in GANs](https://arxiv.org/abs/2209.00698)<br>:star:[code](https://github.com/zikuncshelly/GradCtrl)
* [Studying Bias in GANs through the Lens of Race](https://arxiv.org/abs/2209.02836)
* [FairStyle: Debiasing StyleGAN2 with Style Channel Manipulations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730569.pdf)<br>:house:[project](http://catlab-team.github.io/fairstyle)
* [FingerprintNet: Synthesized Fingerprints for Generated Image Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740071.pdf)
* [Detecting Generated Images by Real Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740089.pdf)<br>:star:[code](https://github.com/Tangsenghenshou/Detecting-Generated-Images-by-Real-Images)
* [High-Fidelity GAN Inversion with Padding Space](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750036.pdf)<br>:house:[project](https://ezioby.github.io/padinv/)
* [A Style-Based GAN Encoder for High Fidelity Reconstruction of Images and Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750579.pdf)<br>:star:[code](https://github.com/InterDigitalInc/FeatureStyleEncoder)
* [BlobGAN: Spatially Disentangled Scene Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750613.pdf)<br>:house:[project](https://dave.ml/blobgan/)
* [GAN with Multivariate Disentangling for Controllable Hair Editing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750653.pdf)<br>:star:[code](https://github.com/XuyangGuo/CtrlHair)
* [StyleGAN-Human: A Data-Centric Odyssey of Human Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760001.pdf)<br>:star:[code](https://github.com/stylegan-human/StyleGAN-Human)
* [EAGAN: Efficient Two-Stage Evolutionary Architecture Search for GANs](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760036.pdf)<br>:star:[code](https://github.com/marsggbo/EAGAN)
* [JoJoGAN: One Shot Face Stylization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760124.pdf)
* [HairNet: Hairstyle Transfer with Pose Changes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760628.pdf)
* [EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760714.pdf)<br>:star:[code](https://github.com/Chenyu-Yang-2000/EleGANt)
* [Editing Out-of-Domain GAN Inversion via Differential Activations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770001.pdf)<br>:star:[code](https://github.com/HaoruiSong622/Editing-Out-of-Domain)
* [On the Robustness of Quality Measures for GANs](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770018.pdf)<br>:star:[code](https://github.com/MotasemAlfarra/R-FID-Robustness-of-Quality-Measures-for-GANs)
* [Diverse Generation from a Single Video Made Possible](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770494.pdf)<br>:house:[project](https://nivha.github.io/vgpnn)
* [Rayleigh EigenDirections (REDs): Nonlinear GAN Latent Space Traversals for Multidimensional Features](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770513.pdf)
* [Generating Natural Images with Direct Patch Distributions Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770547.pdf)<br>:star:[code](https://github.com/ariel415el/GPDM) 
* [TREND: Truncated Generalized Normal Density Estimation of Inception Embeddings for GAN Evaluation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830087.pdf)
* [Neural Scene Decoration from a Single Photograph](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830137.pdf)<br>:star:[code](https://github.com/hkust-vgd/neural_scene_decoration)
* [ChunkyGAN: Real Image Inversion via Segments](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830191.pdf)
* [GAN Cocktail: Mixing GANs without Dataset Access](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830207.pdf)<br>:house:[project](https://omriavrahami.com/GAN-cocktail-page/)
* [DuelGAN: A Duel between Two Discriminators Stabilizes the GAN Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830290.pdf)<br>:star:[code](https://github.com/UCSC-REAL/DuelGAN)
* 线稿上色
  * [Eliminating Gradient Conflict in Reference-based Line-Art Colorization](https://arxiv.org/abs/2207.06095)<br>:star:[code](https://github.com/kunkun0w0/SGA)
* 图像生成
  * [WaveGAN: Frequency-aware GAN for High-Fidelity Few-shot Image Generation](https://arxiv.org/abs/2207.07288)<br>:star:[code](https://github.com/kobeshegu/ECCV2022_WaveGAN)
* GAN逆映射
  * [IntereStyle: Encoding an Interest Region for Robust StyleGAN Inversion](https://arxiv.org/abs/2209.10811)
* 妆发迁移
  * [RamGAN: Region Attentive Morphing GAN for Region-Level Makeup Transfer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820696.pdf)
* 文本消除
  * [Don't Forget Me: Accurate Background Recovery for Text Removal via Modeling Local-Global Context](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880406.pdf)<br>:star:[code](https://github.com/lcy0604/CTRNet)

<a name="16"/>

## 16.Transformer
* [k-means Mask Transformer](https://arxiv.org/abs/2207.04044)<br>:star:[code](https://github.com/google-research/deeplab2)
* [Outpainting by Queries](https://arxiv.org/abs/2207.05312)<br>:star:[code](https://github.com/Kaiseem/QueryOTR)
* [Laplacian Mesh Transformer: Dual Attention and Topology Aware Network for 3D Mesh Classification and Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890532.pdf)   
* [Locality Guidance for Improving Vision Transformers on Tiny Datasets](https://arxiv.org/abs/2207.10026)<br>:star:[code](https://github.com/lkhl/tiny-transformers)
* [ParC-Net: Position Aware Circular Convolution with Merits from ConvNets and Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860600.pdf)<br>:star:[code](https://github.com/hkzhang91/ParC-Net)
* [MTFormer: Multi-task Learning via Transformer and Cross-Task Reasoning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870299.pdf)
* [TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/abs/2207.10666)<br>:star:[code](https://github.com/microsoft/Cream/tree/main/TinyViT)
* [MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis](https://arxiv.org/abs/2207.10228)
* [An Impartial Take to the CNN vs Transformer Robustness Contest](https://arxiv.org/abs/2207.11347)
* [Ghost-free High Dynamic Range Imaging with Context-aware Transformer](https://arxiv.org/abs/2208.05114)<br>:star:[code](https://github.com/megvii-research/HDR-Transformer) 
* [EdgeViTs: Competing Light-Weight CNNs on Mobile Devices with Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710294.pdf)<br>:star:[code](https://github.com/saic-fi/edgevit)
* [Adaptive Token Sampling for Efficient Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710397.pdf)<br>:open_mouth:oral:house:[project](https://adaptivetokensampling.github.io/)
* [Self-Slimmed Vision Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710433.pdf)<br>:star:[code](https://github.com/Sense-X/SiT)
* [Are Vision Transformers Robust to Patch Perturbations?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720399.pdf)
* [Selective TransHDR: Transformer-Based Selective HDR Imaging Using Ghost Region Mask](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770292.pdf)
* [BLT: Bidirectional Layout Transformer for Controllable Layout Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770477.pdf)<br>:house:[project](https://shawnkx.github.io/blt)
* [Convolutional Embedding Makes Hierarchical Vision Transformer Stronger](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800722.pdf)
* [AMixer: Adaptive Weight Mixing for Self-Attention Free Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810051.pdf)<br>:star:[code](https://github.com/raoyongming/AMixer)
* [Doubly-Fused ViT: Fuse Information from Vision Transformer Doubly with Local Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830723.pdf)
* [VIP: Unified Certified Detection and Recovery for Patch Attack with Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850566.pdf)
* [Improving Vision Transformers by Revisiting High-Frequency Components](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840001.pdf)<br>:star:[code](https://github.com/jiawangbai/HAT)
* [VSA: Learning Varied-Size Window Attention in Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850460.pdf)<br>:star:[code](https://github.com/ViTAE-Transformer/ViTAE-VSA)
* [DaViT: Dual Attention Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840073.pdf)<br>:star:[code](https://github.com/microsoft/DaViT)
* [KVT: k-NN Attention for Boosting Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840281.pdf)<br>:star:[code](https://github.com/damo-cv/KVT)
* [ScalableViT: Rethinking the Context-Oriented Generalization of Vision Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840473.pdf)<br>:star:[code](https://github.com/Yangr116/ScalableViT)
* [DeiT III: Revenge of the ViT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840509.pdf)
* [Sliced Recursive Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840716.pdf)<br>:star:[code](https://github.com/szq0214/SReT)
* [Improving Closed and Open-Vocabulary Attribute Prediction Using Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850199.pdf)<br>:house:[project](https://vkhoi.github.io/TAP)
* [Training Vision Transformers with Only 2040 Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850218.pdf)

<a name="15"/>

## 15.Vision-Language(视觉语言)
* [FashionViL: Fashion-Focused Vision-and-Language Representation Learning](https://arxiv.org/abs/2207.08150)<br>:star:[code](https://github.com/BrandonHanx/mmf)
* [NewsStories: Illustrating articles with visual summaries](https://arxiv.org/abs/2207.13061)<br>:star:[code](https://github.com/NewsStoriesData/newsstories.github.io)
* [Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding](https://arxiv.org/abs/2207.14698)<br>:star:[code](https://github.com/haojc/ShufflingVideosForTSG)
* [Frozen CLIP Models are Efficient Video Learners](https://arxiv.org/abs/2208.03550)<br>:star:[code](https://github.com/OpenGVLab/efficient-video-recognition)
* [Generative Negative Text Replay for Continual Vision-Language Pretraining](https://arxiv.org/abs/2210.17322)
* [This Is My Unicorn, Fluffy”: Personalizing Frozen Vision-Language Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800544.pdf)<br>:star:[code](https://github.com/NVlabs/PerVLBenchmark)  
* 视觉表征学习
  * [Wave-ViT: Unifying Wavelet and Transformers for Visual Representation Learning](https://arxiv.org/abs/2207.04978)<br>:star:[code](https://github.com/YehLi/ImageNetModel)
  * [Unsupervised Visual Representation Learning by Synchronous Momentum Grouping](https://arxiv.org/abs/2207.06167)
  * [Learning Visual Representation from Modality-Shared Contrastive Language-Image Pre-training](https://arxiv.org/abs/2207.12661)<br>:star:[code](https://github.com/Hxyou/MSCLIP) 
* VLN
  * [Learning from Unlabeled 3D Environments for Vision-and-Language Navigation](https://arxiv.org/abs/2208.11781)<br>:house:[project](https://cshizhe.github.io/projects/hm3d_autovln.html)
  * [Bridging the visual gap in VLN via semantically richer instructions](https://arxiv.org/abs/2210.15565)
  * [A Dataset for Interactive Vision-Language Navigation with Unknown Command Feasibility](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680304.pdf)
* 视觉重定位
  * [Map-free Visual Relocalization: Metric Pose Relative to a Single Image](https://arxiv.org/abs/2210.05494)

<a name="14"/>

## 14.Visual Answer Questions(视觉问答)
* [Weakly Supervised Grounding for VQA in Vision-Language Transformers](https://arxiv.org/abs/2207.02334)<br>:star:[code](https://github.com/aurooj/WSG-VQA-VLTransformers)
* [Rethinking Data Augmentation for Robust Visual Question Answering](https://arxiv.org/abs/2207.08739)<br>:star:[code](https://github.com/ItemZheng/KDDAug)
* [Video Question Answering with Iterative Video-Text Co-Tokenization](https://arxiv.org/abs/2208.00934)<br>:star:[code](https://sites.google.com/view/videoqa-cotokenization)
* Video-QA
  * [Video Graph Transformer for Video Question Answering](https://arxiv.org/abs/2207.05342)<br>:star:[code](https://github.com/sail-sg/VGT)


<a name="13"/>

## 13.Human-Object Interaction(人物交互)
* [Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection](https://arxiv.org/abs/2207.05293)<br>:star:[code](https://github.com/MuchHair/HQM)
* [Geometric Features Informed Multi-person Human-object Interaction Recognition in Videos](https://arxiv.org/abs/2207.09425)<br>:star:[code](https://github.com/tanqiu98/2G-GCN)
* [IGFormer: Interaction Graph Transformer for Skeleton-based Human Interaction Recognition](https://arxiv.org/abs/2207.12100)
* [Mining Cross-Person Cues for Body-Part Interactiveness Learning in HOI Detection](https://arxiv.org/abs/2207.14192)<br>:star:[code](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness)
* [Iwin: Human-Object Interaction Detection via Transformer with Irregular Windows](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640085.pdf)
* [SAGA: Stochastic Whole-Body Grasping with Contact](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660251.pdf)<br>:house:[project](https://jiahaoplus.github.io/SAGA/saga.html)
* [Chairs Can Be Stood On: Overcoming Object Bias in Human-Object Interaction Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840645.pdf)
* [Discovering Human-Object Interaction Concepts via Self-Compositional Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870454.pdf)<br>:star:[code](https://github.com/zhihou7/HOI-CL)
* 交互式物体分割
  * [Self-Supervised Interactive Object Segmentation Through a Singulation-and-Grasping Approach](https://arxiv.org/abs/2207.09314)<br>:house:[project](https://sites.google.com/umn.edu/sag-interactive-segmentation)
* HOS
  * [Fine-Grained Egocentric Hand-Object Segmentation: Dataset, Model, and Applications](https://arxiv.org/abs/2208.03826)<br>:star:[code](https://github.com/owenzlz/EgoHOS)
* 手物交互
  * [TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630001.pdf)
  * 抓握合成(手物交互)
    * [Grasp'D: Differentiable Contact-Rich Grasp Synthesis for Multi-Fingered Hands](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660197.pdf)<br>:house:[project](https://graspd-eccv22.github.io/)
* 人椅互动
  * [COUCH: Towards Controllable Human-Chair Interactions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650508.pdf)  

<a name="12"/>

## 12.Action Detection(人体动作检测与识别)
* 动作识别
  * [PrivHAR: Recognizing Human Actions from Privacy-Preserving Lens](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640310.pdf)<br>:house:[project](https://carloshinojosa.me/project/privhar/)
  * [Hierarchically Self-Supervised Transformer for Human Skeleton Representation Learning](https://arxiv.org/abs/2207.09644)<br>:star:[code](https://github.com/yuxiaochen1103/Hi-TRS)
  * [An Efficient Spatio-Temporal Pyramid Transformer for Action Detection](https://arxiv.org/abs/2207.10448)
  * [Spatiotemporal Self-attention Modeling with Temporal Patch Shift for Action Recognition](https://arxiv.org/abs/2207.13259)<br>:star:[code](https://github.com/MartinXM/TPS)
  * [Privacy-Preserving Action Recognition via Motion Difference Quantization](https://arxiv.org/abs/2208.02459)<br>:star:[code](https://github.com/suakaw/BDQ_PrivacyAR)
  * [SOS! Self-Supervised Learning over Sets of Handled Objects in Egocentric Action Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730603.pdf)
  * [Real-time Online Video Detection with Temporal Smoothing Transformers](https://arxiv.org/abs/2209.09236)<br>:star:[code](https://github.com/zhaoyue-zephyrus/TeSTra/)
  * [CycDA: Unsupervised Cycle Domain Adaptation to Learn from Image to Video](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630684.pdf)
  * [Uncertainty-Based Spatial-Temporal Attention for Online Action Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640068.pdf)
  * [Is Appearance Free Action Recognition Possible?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640154.pdf)
  * [Panoramic Human Activity Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640242.pdf)
  * [Delving into Details: Synopsis-to-Detail Networks for Video Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640259.pdf)<br>:star:[code](https://github.com/liang4sx/S2DNet)
  * 细粒度动作识别
    * [Dynamic Spatio-Temporal Specialization Learning for Fine-Grained Action Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640381.pdf)
    * [Dynamic Spatio-Temporal Specialization Learning for Fine-Grained Action Recognition](https://arxiv.org/abs/2209.01425)
  * 零样本动作识别
    * [CLASTER: Clustering with Reinforcement Learning for Zero-Shot Action Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800177.pdf)<br>:house:[project](https://sites.google.com/view/claster-zsl/home) 
    * [Rethinking Zero-Shot Action Recognition: Learning from Latent Atomic Actions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640102.pdf)
  * 小样本动作识别
    * [Few-Shot Action Recognition with Hierarchical Matching and Contrastive Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640293.pdf)
    * [Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640172.pdf)
  * 3D动作识别
    * [Collaborating Domain-shared and Target-specific Feature Clustering for Cross-domain 3D Action Recognition](https://arxiv.org/abs/2207.09767)
    * [CMD: Self-supervised 3D Action Representation Learning with Cross-modal Mutual Distillation](https://arxiv.org/abs/2208.12448)<br>:open_mouth:oral:star:[code](https://github.com/maoyunyao/CMD)
    * [Continual 3D Convolutional Neural Networks for Real-Time Processing of Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640364.pdf)<br>:star:[code](https://github.com/lukashedegaard/co3d)
    * [Egocentric Activity Recognition and Localization on a 3D Map](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730620.pdf)
  * 基于骨架动作识别
    * [Global-local Motion Transformer for Unsupervised Skeleton-based Action Learning](https://arxiv.org/abs/2207.06101)<br>:star:[code](https://github.com/Boeun-Kim/GL-Transformer)  
* 社会群体活动识别
  * [Hunting Group Clues with Transformers for Social Group Activity Recognition](https://arxiv.org/abs/2207.05254)
  * [Entry-Flipped Transformer for Inference and Prediction of Participant Behavior](https://arxiv.org/abs/2207.06235)
  * [Hunting Group Clues with Transformers for Social Group Activity Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640018.pdf)
* 时序动作检测
  * [Semi-Supervised Temporal Action Detection with Proposal-Free Masking](https://arxiv.org/abs/2207.07059)<br>:star:[code](https://github.com/sauradip/SPOT)
  * [Temporal Action Detection with Global Segmentation Mask Learning](https://arxiv.org/abs/2207.06580)<br>:star:[code](https://github.com/sauradip/TAGS)
  * [ReAct: Temporal Action Detection with Relational Queries](https://arxiv.org/abs/2207.07097)<br>:star:[code](https://github.com/sssste/React)
  * [Zero-Shot Temporal Action Detection via Vision-Language Prompting](https://arxiv.org/abs/2207.08184)<br>:star:[code](https://github.com/sauradip/STALE)
  * [Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions](https://arxiv.org/abs/2207.11805)<br>:star:[code](https://github.com/lizhi1104/HAAN)
  * [Proposal-Free Temporal Action Detection via Global Segmentation Mask Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630632.pdf)<br>:star:[code](https://github.com/sauradip/TAGS)
* 时序动作定位
  * [Dual-Evidential Learning for Weakly-Supervised Temporal Action Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640190.pdf)<br>:star:[code](github.com/MengyuanChen21/ECCV2022-DELU)
  * [ActionFormer: Localizing Moments of Actions with Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf)<br>:star:[code](https://github.com/happyharrycn/actionformer_release)
* 时序动作分割
  * [Unified Fully and Timestamp Supervised Temporal Action Segmentation via Sequence to Sequence Translation](https://arxiv.org/abs/2209.00638)
* Action Quality Assessment(行动质量评估)
  * [Action Quality Assessment with Temporal Parsing Transformer](https://arxiv.org/abs/2207.09270)
  * [Pairwise Contrastive Learning Network for Action Quality Assessment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640450.pdf)<br>:star:[code](https://github.com/hqu-cst-mmc/PCLN)
  * [A Generalized & Robust Framework For Timestamp Supervision in Temporal Action Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640276.pdf)

<a name="11"/>

## 11.Video  
* [Dynamic Temporal Filtering in Video Models](https://arxiv.org/abs/2211.08252)<br>:star:[code](https://github.com/FuchenUSTC/DTF)
* 视频合成
  * [Layered Controllable Video Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760523.pdf)<br>:house:[project](https://gabriel-huang.github.io/layered_controllable_video_generation/)
  * [Sound-Guided Semantic Video Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770034.pdf)<br>:house:[project](https://kuai-lab.github.io/eccv2022sound/)
  * [Controllable Video Generation through Global and Local Motion Dynamics](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770069.pdf)<br>:house:[project](https://araachie.github.io/glass/)
  * [Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770103.pdf)<br>:house:[project](https://songweige.github.io/projects/tats/)
* 视频-视频合成
  * [Fast-Vid2Vid: Spatial-Temporal Compression for Video-to-Video Synthesis](https://arxiv.org/abs/2207.05049)<br>:star:[code](https://github.com/fast-vid2vid/fast-vid2vid):house:[project](https://fast-vid2vid.github.io/)
* 视频帧插值
  * [A Perceptual Quality Metric for Video Frame Interpolation](https://arxiv.org/abs/2210.01879)<br>:star:[code](https://github.com/hqqxyy/VFIPS)
* 视频生成
  * [RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos](https://arxiv.org/abs/2207.11075)<br>:open_mouth:oral:star:[code](https://github.com/megvii-research/RealFlow)
* 视频质量评估
  * [FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling](https://arxiv.org/abs/2207.02595)<br>:star:[code](https://github.com/timothyhtimothy/FAST-VQA)
  * [Telepresence Video Quality Assessment](https://arxiv.org/abs/2207.09956)
* 视频修复
  * [Error Compensation Framework for Flow-Guided Video Inpainting](https://arxiv.org/abs/2207.10391)
  * [Flow-Guided Transformer for Video Inpainting](https://arxiv.org/abs/2208.06768)<br>:star:[code](https://github.com/hitachinsk/FGT)
  * [Video Restoration Framework and Its Meta-Adaptations to Data-Poor Conditions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880142.pdf)<br>:star:[code](https://github.com/pwp1208/Meta_Video_Restoration)
* 视频去模糊
  * [Spatio-Temporal Deformable Attention Network for Video Deblurring](https://arxiv.org/abs/2207.10852)<br>:star:[code](https://github.com/huicongzhang/STDAN):house:[project](https://vilab.hit.edu.cn/projects/stdan/)
  * [Efficient Video Deblurring Guided by Motion Magnitude](https://arxiv.org/abs/2207.13374)<br>:star:[code](https://github.com/sollynoay/MMP-RNN)
* 视频对话
  * [Video Dialog as Conversation about Objects Living in Space-Time](https://arxiv.org/abs/2207.03656)<br>:star:[code](https://github.com/hoanganhpham1006/COST)
* 有源扬声器检测(视频会议)
  * [Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection](https://arxiv.org/abs/2207.07783)<br>:star:[code](https://github.com/SRA2/SPELL)
* VOS
  * [XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://arxiv.org/abs/2207.07115)<br>:star:[code](https://github.com/hkchengrex/XMem):house:[project](https://hkchengrex.github.io/XMem/):tv:[video](https://youtu.be/mwOP8l3zVNw)
  * [Tackling Background Distraction in Video Object Segmentation](https://arxiv.org/abs/2207.06953)<br>:star:[code](https://github.com/suhwan-cho/TBD)
  * [BATMAN: Bilateral Attention Transformer in Motion-Appearance Neighboring Space for Video Object Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890603.pdf)
  * [Hierarchical Feature Alignment Network for Unsupervised Video Object Segmentation](https://arxiv.org/abs/2207.08485)<br>:star:[code](https://github.com/NUST-Machine-Intelligence-Laboratory/HFAN)
  * [Learning Quality-aware Dynamic Memory for Video Object Segmentation](https://arxiv.org/abs/2207.07922)<br>:star:[code](https://github.com/workforai/QDMN)
  * [Global Spectral Filter Memory Network for Video Object Segmentation](https://arxiv.org/abs/2210.05567)<br>:star:[code](https://github.com/workforai/GSFM)
* VIS
  * [In Defense of Online Models for Video Instance Segmentation](https://arxiv.org/abs/2207.10661)<br>:open_mouth:oral:star:[code](https://github.com/wjf5203/VNext)
  * [Instance As Identity: A Generic Online Paradigm for Video Instance Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890515.pdf)<br>:star:[code](https://github.com/zfonemore/IAI)
  * [Video Instance Segmentation via Multi-Scale Spatio-Temporal Split Attention Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890657.pdf)
  * [Video Mask Transfiner for High-Quality Video Instance Segmentation](https://arxiv.org/abs/2207.14012)
  * [SeqFormer: Sequential Transformer for Video Instance Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880547.pdf)<br>:star:[code](https://github.com/wjf5203/SeqFormer)
* VSS
  * [Mining Relations among Cross-Frame Affinities for Video Semantic Segmentation](https://arxiv.org/abs/2207.10436)<br>:star:[code](https://github.com/GuoleiSun/VSS-MRCFA)
  * [Domain Adaptive Video Segmentation via Temporal Pseudo Supervision](https://arxiv.org/abs/2207.02372)<br>:star:[code](https://github.com/xing0047/TPS)
  * [Is It Necessary to Transfer Temporal Knowledge for Domain Adaptive Video Semantic Segmentation?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870351.pdf)<br>:star:[code](https://github.com/W-zx-Y/I2VDA)
* VPS
  * [Waymo Open Dataset: Panoramic Video Panoptic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890052.pdf)<br>:house:[project](https://waymo.com/open/)
  * [PolyphonicFormer: Unified Query Learning for Depth-Aware Video Panoptic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870574.pdf)<br>:star:[code](https://github.com/HarborYuan/PolyphonicFormer)
* 视频抠图
  * [One-Trimap Video Matting](https://arxiv.org/abs/2207.13353)<br>:star:[code](https://github.com/Hongje/OTVM):tv:[video](https://www.youtube.com/watch?v=qkda4fHSyQE)
* 视频表征
  * [E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context](https://arxiv.org/abs/2207.08132)<br>:star:[code](https://github.com/kyleleey/E-NeRV) 
  * [Static and Dynamic Concepts for Self-supervised Video Representation Learning](https://arxiv.org/abs/2207.12795)
* 视频传输
  * [Efficient Meta-Tuning for Content-aware Neural Video Delivery](https://arxiv.org/abs/2207.09691)<br>:star:[code](https://github.com/Neural-video-delivery/EMT-Pytorch-ECCV2022)
* 运动分割
  * [ParticleSfM: Exploiting Dense Point Trajectories for Localizing Moving Cameras in the Wild](https://arxiv.org/abs/2207.09137)<br>:star:[code](https://github.com/bytedance/particle-sfm):house:[project](http://b1ueber2y.me/projects/ParticleSfM/)
* 视频异常检测
  * [Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles](https://arxiv.org/abs/2207.10172)<br>:star:[code](https://github.com/wizyoung/YOLOv3)
  * [Towards Open Set Video Anomaly Detection](https://arxiv.org/abs/2208.11113)
  * [Scale-Aware Spatio-Temporal Relation Learning for Video Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640328.pdf)<br>:star:[code](https://github.com/nutuniv/SSRL)
  * [Self-Supervised Sparse Representation for Video Anomaly Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730727.pdf)<br>:star:[code](https://github.com/louisYen/S3R)
* 视频识别
  * [Temporal Saliency Query Network for Efficient Video Recognition](https://arxiv.org/abs/2207.10379)<br>:house:[project](https://lawrencexia2008.github.io/projects/tsqnet)
  * [NSNet: Non-saliency Suppression Sampler for Efficient Video Recognition](https://arxiv.org/abs/2207.10388)<br>:house:[project](https://lawrencexia2008.github.io/projects/nsnet)
  * [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816)<br>:open_mouth:oral:star:[code](https://github.com/microsoft/VideoX/tree/master/X-CLIP)
  * [AdaFocusV3: On Unified Spatial-temporal Dynamic Video Recognition](https://arxiv.org/abs/2209.13465)
* 视频理解
  * [Spotting Temporally Precise, Fine-Grained Events in Video](https://arxiv.org/abs/2207.10213)<br>:star:[code](https://github.com/jhong93/spot):house:[project](https://jhong93.github.io/projects/spot.html)
  * [Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding](https://arxiv.org/abs/2208.00281)
  * [Panoramic Vision Transformer for Saliency Detection in 360° Videos](https://arxiv.org/abs/2209.08956)<br>:star:[code](https://github.com/hs-yn/PAVER)
  * [Streaming Multiscale Deep Equilibrium Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710189.pdf)<br>:house:[project](https://ufukertenli.github.io/streamdeq/)
  * [Learning Shadow Correspondence for Video Shadow Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770709.pdf)
* 视频分类
  * [Inductive and Transductive Few-Shot Video Classification via Appearance and Temporal Alignments](https://arxiv.org/abs/2207.10785)<br>:star:[code](https://github.com/VinAIResearch/fsvc-ata)
* 视频卷帘快门(Rolling shutter)
  * [Combining Internal and External Constraints for Unrolling Shutter in Videos](https://arxiv.org/abs/2207.11725)
* Video Transition Effects(视频转场特效)
  * [AutoTransition: Learning to Recommend Video Transition Effects](https://arxiv.org/abs/2207.13479)<br>:star:[code](https://github.com/acherstyx/AutoTransition)
* 图像-视频编解码
  * [AlphaVC: High-Performance and Efficient Learned Video Compression](https://arxiv.org/abs/2207.14678) 
  * [CANF-VC: Conditional Augmented Normalizing Flows for Video Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760193.pdf)<br>:star:[code](https://github.com/NYCU-MAPL/CANF-VC)
  * [Expanded Adaptive Scaling Normalization for End to End Image Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770392.pdf)
  * [Coarse-to-Fine Sparse Transformer for Hyperspectral Image Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770690.pdf)<br>:star:[code](https://github.com/caiyuanhao1998/MST)
  * [Content Adaptive Latents and Decoder for Neural Image Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780545.pdf)
  * [Contextformer: A Transformer with Spatio-Channel Attention for Context Modeling in Learned Image Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790436.pdf)
  * [RAWtoBit: A Fully End-to-End Camera ISP Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790487.pdf)
  * [Content-Oriented Learned Image Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790621.pdf)
  * [Implicit Neural Representations for Image Compression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860073.pdf)
  * [Neural Video Compression Using GANs for Detail Synthesis and Propagation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860549.pdf)
* 视频摘要
  * [TL;DW? Summarizing Instructional Videos with Task Relevance & Cross-Modal Saliency](https://arxiv.org/abs/2208.06773)<br>:star:[code](https://github.com/medhini/Instructional-Video-Summarization):house:[project](https://medhini.github.io/ivsum/)
* Video Grounding
  * [Graph2Vid: Flow graph to Video Grounding forWeakly-supervised Multi-Step Localization](https://arxiv.org/abs/2210.04996)<br>:open_mouth:oral 
* 帧插值
  * [FILM: Frame Interpolation for Large Motion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670244.pdf)<br>:house:[project](https://film-net.github.io/)  
  * [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740608.pdf)<br>:star:[code](https://github.com/megvii-research/ECCV2022-RIFE)
  * [Deep Bayesian Video Frame Interpolation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750141.pdf)<br>:star:[code](https://github.com/Oceanlib/DBVI)
  * [Improving the Perceptual Quality of 2D Animation Interpolation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770275.pdf)
* 视频分析
  * [Event Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710276.pdf) 
* 视频编辑
  * [Temporally Consistent Semantic Video Editing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750355.pdf)
* 视频增强
  * [Learning Cross-Video Neural Representations for High-Quality Frame Interpolation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750509.pdf)<br>:house:[project](https://cigroup.wustl.edu/)
* 视频目标重识别
  * [CAViT: Contextual Alignment Vision Transformer for Video Object Re-identification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740535.pdf)<br>:star:[code](https://github.com/KimWu1994/CAViT)
* 图像视频编辑
  * [Text2LIVE: Text-Driven Layered Image and Video Editing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750705.pdf)
* 视频升格
  * [Learning Spatio-Temporal Downsampling for Effective Video Upscaling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780159.pdf)
* 视频色彩传播
  * [Learned Variational Video Color Propagation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830497.pdf)<br>:star:[code](https://github.com/VLOGroup/LVVCP) 

<a name="10"/>

## 10.Pose Estimation(物体姿势估计)
* 物体姿势
  * [Neural Correspondence Field for Object Pose Estimation](https://arxiv.org/abs/2208.00113)<br>:star:[code](https://github.com/LinHuang17/NCF-code):house:[project](https://linhuang17.github.io/NCF/)
  * [Fusing Local Similarities for Retrieval-based 3D Orientation Estimation of Unseen Objects](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610106.pdf)<br>:star:[code](https://github.com/sailor-z/Unseen_Object_Pose):house:[project](https://sailor-z.github.io/projects/Unseen_Object_Pose.html)
  * [A Visual Navigation Perspective for Category-Level Object Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660121.pdf)<br>:star:[code](https://github.com/wrld/visual_navigation_pose_estimation)
  * [Polarimetric Pose Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690726.pdf)
  * [RayTran: 3D Pose Estimation and Shape Reconstruction of Multiple Objects from Videos with Ray-Traced Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700209.pdf)
* 物体姿势变换
  * [General Object Pose Transformation Network from Unpaired Data](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660286.pdf)
* 抓取物体姿势估计
  * [TransGrasp: Grasp Pose Estimation of a Category of Objects by Transferring Grasps from Only One Labeled Instance](https://arxiv.org/abs/2207.07861)<br>:star:[code](https://github.com/yanjh97/TransGrasp)
* 4D
  * [HuMMan: Multi-modal 4D Human Dataset for Versatile Sensing and Modeling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670549.pdf)<br>:house:[project](https://caizhongang.github.io/projects/HuMMan/)
* 6D
  * [Category-Level 6D Object Pose and Size Estimation using Self-Supervised Deep Prior Deformation Networks](https://arxiv.org/abs/2207.05444)<br>:star:[code](https://github.com/JiehongLin/Self-DPDN)
  * [ShAPO: Implicit Representations for Multi-Object Shape, Appearance, and Pose Optimization](https://arxiv.org/abs/2207.13691)<br>:house:[project](https://zubair-irshad.github.io/projects/ShAPO.html)
  * [RBP-Pose: Residual Bounding Box Projection for Category-Level Pose Estimation](https://arxiv.org/abs/2208.00237)<br>:star:[code](https://github.com/lolrudy/RBP_Pose)
  * [Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features](https://arxiv.org/abs/2209.05624)
  * [Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610018.pdf)<br>:star:[code](https://github.com/Dangzheng/MatchNorm)
  * [Perspective Flow Aggregation for Data-Limited 6D Object Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620087.pdf)<br>:star:[code](https://github.com/cvlab-epfl/perspective-flow-aggregation)
  * [Object Level Depth Reconstruction for Category Level 6D Object Pose Estimation from Monocular RGB Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620212.pdf)
  * [DProST: Dynamic Projective Spatial Transformer Network for 6D Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660357.pdf)<br>:star:[code](https://github.com/parkjaewoo0611/DProST)
  * [WeLSA: Learning to Predict 6D Pose from Weakly Labeled Data Using Shape Alignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680633.pdf)
  * [DCL-Net: Deep Correspondence Learning Network for 6D Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690362.pdf)<br>:star:[code](https://github.com/Gorilla-Lab-SCUT/DCL-Net)
  * [DISP6D: Disentangled Implicit Shape and Pose Learning for Scalable 6D Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690397.pdf)<br>:star:[code](https://github.com/fylwen/DISP-6D)
  * [Vote from the Center: 6 DoF Pose Estimation in RGB-D Images by Radial Keypoint Voting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700331.pdf)<br>:star:[code](https://github.com/aaronwool/rcvpose)
* 9D
  * [CATRE: Iterative Point Clouds Alignment for Category-level Object Pose Refinement](https://arxiv.org/abs/2207.08082)<br>:star:[code](https://github.com/THU-DA-6D-Pose-Group/CATRE)
  
<a name="9"/>

## 9.Human Pose Estimation(人体姿态估计)
* [Self-Constrained Inference Optimization on Structural Groups for Human Pose Estimation](https://arxiv.org/abs/2207.02425)
* [Pose for Everything: Towards Category-Agnostic Pose Estimation](https://arxiv.org/abs/2207.10387)<br>:open_mouth:oral:star:[code](https://github.com/luminxu/Pose-for-Everything)
* [PoseTrans: A Simple Yet Effective Pose Transformation Augmentation for Human Pose Estimation](https://arxiv.org/abs/2208.07755)
* [Learning Visibility for Robust Dense Human Body Estimation](https://arxiv.org/abs/2208.10652)<br>:star:[code](https://github.com/chhankyao/visdb)
* [D&D: Learning Human Dynamics from Dynamic Camera](https://arxiv.org/abs/2209.08790)<br>:open_mouth:oral:star:[code](https://github.com/Jeff-sjtu/DnD)
* [PPT: token-Pruned Pose Transformer for monocular and multi-view human pose estimation](https://arxiv.org/abs/2209.08194)<br>:star:[code](https://github.com/HowieMa/PPT)
* [DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650597.pdf)<br>:star:[code](https://github.com/cure-lab/DeciWatch)
* [SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650615.pdf)<br>:star:[code](https://github.com/cure-lab/SmoothNet)
* [Poseur: Direct Human Pose Regression with Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660071.pdf)<br>:star:[code](https://github.com/aim-uofa/Poseur)
* [SimCC: A Simple Coordinate Classification Perspective for Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660088.pdf)<br>:star:[code](https://github.com/leeyegy/SimCC)
* [Regularizing Vector Embedding in Bottom-Up Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660105.pdf)<br>:star:[code](https://github.com/CR320/CoupledEmbedding)
* [Hallucinating Pose-Compatible Scenes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760487.pdf)
* 运动捕捉
  * [TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts](https://arxiv.org/abs/2207.01696)<br>:star:[code](https://github.com/EricGuo5513/TM2T):house:[project](https://ericguo5513.github.io/TM2T/)
  * [HULC: 3D HUman Motion Capture with Pose Manifold SampLing and Dense Contact Guidance](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820503.pdf)<br>:house:[project](https://vcai.mpi-inf.mpg.de/projects/HULC/)
* 基于点的衣着人体建模
  * [Learning Implicit Templates for Point-Based Clothed Human Modeling](https://arxiv.org/abs/2207.06955)<br>:star:[code](https://github.com/jsnln/fite):house:[project](https://jsnln.github.io/fite/)
* 动态人体数字化
  * [NDF: Neural Deformable Fields for Dynamic Human Modelling](https://arxiv.org/abs/2207.09193)<br>:star:[code](https://github.com/HKBU-VSComputing/2022_ECCV_NDF)
* 人体姿势与形状估计
  * [CLIFF: Carrying Location Information in Full Frames into Human Pose and Shape Estimation](https://arxiv.org/abs/2208.00571)<br>:open_mouth:oral:star:[code](https://github.com/huawei-noah/noah-research/tree/master/CLIFF)
  * [Super-Resolution 3D Human Shape from a Single Low-Resolution Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620435.pdf)<br>:star:[code](https://github.com/marcopesavento/Super-resolution-3D-Human-Shape-from-a-Single-Low-Resolution-Image):house:[project](https://marcopesavento.github.io/SuRS/)
* 三维人体姿势估计
  * [DH-AUG: DH Forward Kinematics Model Driven Augmentation for 3D Human Pose Estimation](https://arxiv.org/abs/2207.09303)<br>:star:[code](https://github.com/hlz0606/DH-AUG-DH-Forward-Kinematics-Model-Driven-Augmentation-for-3D-Human-Pose-Estimation)
  * [Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection](https://arxiv.org/abs/2207.10955)
  * [Explicit Occlusion Reasoning for Multi-person 3D Human Pose Estimation](https://arxiv.org/abs/2208.00090)<br>:star:[code](https://github.com/qihao067/HUPOR)
  * [PoseScript: 3D Human Poses from Natural Language](https://arxiv.org/abs/2210.11795)<br>:house:[project](https://europe.naverlabs.com/research/computer-vision/posescript/)
  * [Multi-Person 3D Pose and Shape Estimation via Inverse Kinematics and Refinement](https://arxiv.org/abs/2210.13529)
  * [3D Human Pose Estimation Using Möbius Graph Convolutional Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610158.pdf)
  * [P-STMO: Pre-trained Spatial Temporal Many-to-One Model for 3D Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650453.pdf)<br>:star:[code](https://github.com/paTRICK-swk/P-STMO)
  * [C3P: Cross-Domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650544.pdf)<br>:star:[code](https://github.com/wucunlin/C3P)
  * [Structural Triangulation: A Closed-Form Solution to Constrained 3D Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650685.pdf)<br>:star:[code](https://github.com/chzh9311/structural-triangulation)
  * [VirtualPose: Learning Generalizable 3D Human Pose Models from Virtual Data](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660054.pdf)<br>:star:[code](https://github.com/wkom/VirtualPose)
  * [Learning to Fit Morphable Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660156.pdf)
  * [EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660176.pdf)<br>:house:[project](https://sanweiliti.github.io/egobody/egobody.html)
  * [AutoAvatar: Autoregressive Neural Fields for Dynamic Avatar Modeling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660216.pdf)<br>:house:[project](https://zqbai-jeremy.github.io/autoavatar)
* Mul-Pose
  * [Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660036.pdf)<br>:star:[code](https://github.com/wmcnally/kapao)
* 三维人体重建
  * [3D Clothed Human Reconstruction in the Wild](https://arxiv.org/abs/2207.10053)<br>:star:[code](https://github.com/hygenie1228/ClothWild_RELEASE)
  * [UNIF: United Neural Implicit Functions for Clothed Human Reconstruction and Animation](https://arxiv.org/abs/2207.09835)<br>:star:[code](https://github.com/ShenhanQian/UNIF)
  * [The One Where They Reconstructed 3D Humans and Environments in TV Shows](https://arxiv.org/abs/2207.14279)<br>:star:[code](https://github.com/ethanweber/sitcoms3D):house:[project](http://ethanweber.me/sitcoms3D/)
  * [Neural Capture of Animatable 3D Human from Monocular Video](https://arxiv.org/abs/2208.08728)  
  * [SUPR: A Sparse Unified Part-Based Human Representation](https://arxiv.org/abs/2210.13861)<br>:star:[code](https://github.com/ahmedosman/SUPR):house:[project](https://supr.is.tue.mpg.de/)
  * [IntegratedPIFu: Integrated Pixel Aligned Implicit Function for Single-view Human Reconstruction](https://arxiv.org/abs/2211.07955)<br>:star:[code](https://github.com/kcyt/IntegratedPIFu)
  * [Learned Vertex Descent:A New Direction for 3D Human Model Fitting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620141.pdf)<br>:star:[code](https://github.com/enriccorona/LVD):house:[project](https://www.iri.upc.edu/people/ecorona/lvd/)
* 三维交互式手部姿势估计
  * [3D Interacting Hand Pose Estimation by Hand De-occlusion and Removal](https://arxiv.org/abs/2207.11061)<br>:star:[code](https://github.com/MengHao666/HDR)
  * [S2Contact: Graph-based Network for 3D Hand-Object Contact Estimation with Semi-Supervised Learning](https://arxiv.org/abs/2208.00874)<br>:star:[code](https://github.com/eldentse/s2contact):house:[project](https://eldentse.github.io/s2contact/)
* 姿势合成
  * [TIPS: Text-Induced Pose Synthesis](https://arxiv.org/abs/2207.11718)<br>:star:[code](https://github.com/prasunroy/tips):house:[project](https://prasunroy.github.io/tips/)
* 手物重建
  * [AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction](https://arxiv.org/abs/2207.12909)<br>:star:[code](https://github.com/zerchen/alignsdf):house:[project](https://zerchen.github.io/projects/alignsdf.html)
* 人体与场景的交互
  * [Compositional Human-Scene Interaction Synthesis with Semantic Control](https://arxiv.org/abs/2207.12824)<br>:star:[code](https://github.com/zkf1997/COINS)
* 人体姿势建模
  * [Pose-NDF: Modeling Human Pose Manifolds with Neural Distance Fields](https://arxiv.org/abs/2207.13807)<br>:open_mouth:oral:house:[project](https://virtualhumans.mpi-inf.mpg.de/posendf/)
* 姿势跟踪
  * [AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing](https://arxiv.org/abs/2207.13784)<br>:star:[code](https://github.com/eth-siplab/AvatarPoser)
* 三维人体网格恢复
  * [Cross-Attention of Disentangled Modalities for 3D Human Mesh Recovery with Transformers](https://arxiv.org/abs/2207.13820)<br>:star:[code](https://github.com/postech-ami/FastMETRO) 
* 三维人体运动预测与生成
  * [Skeleton-Parted Graph Scattering Networks for 3D Human Motion Prediction](https://arxiv.org/abs/2208.00368)<br>:star:[code](https://github.com/MediaBrain-SJTU/SPGSN) 
  * [PoseGPT: Quantization-based 3D Human Motion Generation and Forecasting](https://arxiv.org/abs/2210.10542)<br>:house:[project](https://europe.naverlabs.com/research/computer-vision/posegpt/)
* 姿势迁移
  * [Skeleton-free Pose Transfer for Stylized 3D Characters](https://arxiv.org/abs/2208.00790)<br>:star:[code](https://github.com/zycliao/skeleton-free-pose-transfer):house:[project](https://zycliao.com/sfpt/)
  * [Cross Attention Based Style Distribution for Controllable Person Image Synthesis](https://arxiv.org/abs/2208.00712)<br>:star:[code](https://github.com/xyzhouo/CASD)
* 人体姿势预测
  * [Pose Forecasting in Industrial Human-Robot Collaboration](https://arxiv.org/abs/2208.07308)<br>:star:[code](https://github.com/AlessioSam/CHICO-PoseForecasting)
* 4D
  * [LoRD: Local 4D Implicit Representation for High-Fidelity Dynamic Human Modeling](https://arxiv.org/abs/2208.08622)<br>:star:[code](https://github.com/BoyanJIANG/LoRD):house:[project](https://boyanjiang.github.io/LoRD/)
* 人体网格恢复
  * [Self-supervised Human Mesh Recovery with Cross-Representation Alignment](https://arxiv.org/abs/2209.04596)
* 手部网格估计
  * [Identity-Aware Hand Mesh Estimation and Personalization from RGB Images](https://arxiv.org/abs/2209.10840)<br>:star:[code](https://github.com/deyingk/PersonalizedHandMeshEstimation) 
* 头部网格重建
  * [Realistic One-Shot Mesh-Based Head Avatars](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620336.pdf)<br>:house:[project](https://samsunglabs.github.io/rome/)  
* 人体网格动画
  * [CLIP-Actor: Text-Driven Recommendation and Stylization for Animating Human Meshes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630176.pdf)  
* 音频驱动的风格化手势生成
  * [Audio-Driven Stylized Gesture Generation with Flow-Based Model](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650701.pdf) 

<a name="8"/>

## 8.3D(三维视觉)
* [DeepPS2: Revisiting Photometric Stereo Using Two Differently Illuminated Images](https://arxiv.org/abs/2207.02025)<br>:star:[code](https://github.com/ashisht96/DeepPS2)
* [Towards High-Fidelity Single-view Holistic Reconstruction of Indoor Scenes](https://arxiv.org/abs/2207.08656)<br>:star:[code](https://github.com/UncleMEDM/InstPIFu)
* [Self-calibrating Photometric Stereo by Neural Inverse Rendering](https://arxiv.org/abs/2207.07815)<br>:star:[code](https://github.com/junxuan-li/SCPS-NIR)
* [3DG-STFM: 3D Geometric Guided Student-Teacher Feature Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880124.pdf)<br>:star:[code](https://github.com/Ryan-prime/3DG-STFM) 
* MVS
  * MVPS
    * [PS-NeRF: Neural Inverse Rendering for Multi-view Photometric Stereo](https://arxiv.org/abs/2207.11406)<br>:house:[project](https://ywq.github.io/psnerf/)
* 3D场景合成
  * [Simple and Effective Synthesis of Indoor 3D Scenes](https://arxiv.org/abs/2204.02960)<br>:tv:[video](https://www.youtube.com/watch?v=lhwwlrRfFp0)
* 场景重建
  * [Initialization and Alignment for Adversarial Texture Optimization](https://arxiv.org/abs/2207.14289)<br>:star:[code](https://github.com/Xiaoming-Zhao/advtex_init_align):house:[project](https://xiaoming-zhao.github.io/projects/advtex_init_align/)
* 深度估计
  * [Physical Attack on Monocular Depth Estimation with Optimal Adversarial Patches](https://arxiv.org/abs/2207.04718)
  * [Towards Scale-Aware, Robust, and Generalizable Unsupervised Monocular Depth Estimation by Integrating IMU Motion Dynamics](https://arxiv.org/abs/2207.04680)<br>:star:[code](https://github.com/SenZHANG-GitHub/ekf-imu-depth)
  * [Stereo Depth Estimation with Echoes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870489.pdf)
  * [JPerceiver: Joint Perception Network for Depth, Pose and Layout Estimation in Driving Scenes](https://arxiv.org/abs/2207.07895)<br>:star:[code](https://github.com/sunnyHelen/JPerceiver)
  * [RA-Depth: Resolution Adaptive Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/2207.11984)<br>:star:[code](https://github.com/hmhemu/RA-Depth)
  * [PointFix: Learning to Fix Domain Bias for Robust Online Stereo Adaptation](https://arxiv.org/abs/2207.13340)
  * [Depth Field Networks for Generalizable Multi-view Scene Representation](https://arxiv.org/abs/2207.14287)<br>:house:[project](https://sites.google.com/view/tri-define)
  * [Gradient-based Uncertainty for Monocular Depth Estimation](https://arxiv.org/abs/2208.02005)<br>:star:[code](https://github.com/jhornauer/GrUMoDepth)
  * [DevNet: Self-supervised Monocular Depth Learning via Density Volume Construction](https://arxiv.org/abs/2209.06351)<br>:star:[code](https://github.com/gitkaichenzhou/DevNet)
  * [Self-distilled Feature Aggregation for Self-supervised Monocular Depth Estimation](https://arxiv.org/abs/2209.07088)<br>:star:[code](https://github.com/ZM-Zhou/SDFA-Net_pytorch)
  * [3D-PL: Domain Adaptive Depth Estimation with 3D-aware Pseudo-Labeling](https://arxiv.org/abs/2209.09231)<br>:star:[code](https://github.com/ccc870206/3D-PL)
  * [DELTAR: Depth Estimation from a Light-weight ToF Sensor and RGB Image](https://arxiv.org/abs/2209.13362)<br>:star:[code](https://github.com/zju3dv/deltar):house:[project](https://zju3dv.github.io/deltar/)
  * [FloatingFusion: Depth from ToF and Image-stabilized Stereo Cameras](https://arxiv.org/abs/2210.02785)
  * [Context-Enhanced Stereo Transformer](https://arxiv.org/abs/2210.11719)
  * [Adaptive Co-Teaching for Unsupervised Monocular Depth Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610089.pdf)<br>:star:[code](https://github.com/Mkalilia/MUSTNet)
  * [PanoFormer: Panorama Transformer for Indoor 360° Depth Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610193.pdf)<br>:star:[code](https://github.com/zhijieshen-bjtu/PanoFormer)
  * [Towards Comprehensive Representation Enhancement in Semantics-Guided Self-Supervised Monocular Depth Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610299.pdf)
  * [LocalBins: Improving Depth Estimation by Learning Local Distributions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610473.pdf)<br>:star:[code](https://github.com/shariqfarooq123/LocalBins)
  * [Depth Map Decomposition for Monocular Depth Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620018.pdf)
  * [Uncertainty Quantification in Depth Estimation via Constrained Ordinal Regression](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620229.pdf)<br>:star:[code](https://github.com/timmy11hu/ConOR)
  * [Spike Transformer: Monocular Depth Estimation for Spiking Camera](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670034.pdf)<br>:star:[code](https://github.com/Leozhangjiyuan/MDE-SpikingCamera)
  * [Learning Phase Mask for Privacy-Preserving Passive Depth Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670497.pdf)
* 深度补全
  * [GraphCSPN: Geometry-Aware Depth Completion via Dynamic GCNs](https://arxiv.org/abs/2210.10758)<br>:star:[code](https://github.com/xinliu20/GraphCSPN_ECCV2022)
  * [RigNet: Repetitive Image Guided Network for Depth Completion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870211.pdf)
  * [Multi-modal Masked Pre-training for Monocular Panoramic Depth Completion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610372.pdf)
  * [Monitored Distillation for Positive Congruent Depth Completion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620035.pdf)<br>:star:[code](https://github.com/alexklwong/mondi-python)
  * [CostDCNet: Cost Volume Based Depth Completion for a Single RGB-D Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620248.pdf)<br>:star:[code](https://github.com/kamse/CostDCNet)
* 三维视觉
  * [A Closer Look at Invariances in Self-supervised Pre-training for 3D Vision](https://arxiv.org/abs/2207.04997)
  * [Neural Density-Distance Fields](https://arxiv.org/abs/2207.14455)<br>:star:[code](https://github.com/ueda0319/neddf)
  * [DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620104.pdf)<br>:house:[project](https://lemonatsu.github.io/danbo)
  * [CryoAI: Amortized Inference of Poses for Ab Initio Reconstruction of 3D Molecular Volumes from Real Cryo-EM Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810533.pdf)
* 三维房间布局
  * [3D Room Layout Estimation from a Cubemap of Panorama Image via Deep Manhattan Hough Transform](https://arxiv.org/abs/2207.09291)<br>:star:[code](https://github.com/Starrah/DMH-Net)
* 三维重建
  * [Object-Compositional Neural Implicit Surfaces](https://arxiv.org/abs/2207.09686)<br>:star:[code](https://github.com/QianyiWu/objsdf):house:[project](https://wuqianyi.top/objectsdf/):tv:[video](https://www.youtube.com/watch?v=23vxOV19bEw&feature=youtu.be)
  * [Perspective Phase Angle Model for Polarimetric 3D Reconstruction](https://arxiv.org/abs/2207.09629)<br>:star:[code](https://github.com/GCChen97/ppa4p3d)  
  * [Monocular 3D Object Reconstruction with GAN Inversion](https://arxiv.org/abs/2207.10061)<br>:star:[code](https://github.com/junzhezhang/mesh-inversion):house:[project](https://www.mmlab-ntu.com/project/meshinversion/)
  * [Structural Causal 3D Reconstruction](https://arxiv.org/abs/2207.10156)
  * [2D GANs Meet Unsupervised Single-view 3D Reconstruction](https://arxiv.org/abs/2207.10183)<br>:star:[code](https://github.com/liuf1990/GANSVR):house:[project](http://cvlab.cse.msu.edu/project-gansvr.html)
  * [Few-shot Single-view 3D Reconstruction with Memory Prior Contrastive Network](https://arxiv.org/abs/2208.00183)
  * [PlaneFormers: From Sparse View Planes to 3D Reconstruction](https://arxiv.org/abs/2208.04307)<br>:star:[code](https://github.com/samiragarwala/PlaneFormers):house:[project](https://samiragarwala.github.io/PlaneFormers/):tv:[video](https://youtu.be/3VPsOxXEMlI)
  * [SimpleRecon: 3D Reconstruction Without 3D Convolutions](https://arxiv.org/abs/2208.14743)<br>:star:[code](https://nianticlabs.github.io/simplerecon/)
  * [Share with Thy Neighbors: Single-View Reconstruction by Cross-Instance Consistency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610282.pdf)
  * [SketchSampler: Sketch-Based 3D Reconstruction via View-Dependent Depth Sampling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610457.pdf)
  * [Semi-Supervised Single-View 3D Reconstruction via Prototype Shape Priors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610528.pdf)<br>:star:[code](https://github.com/ChenHsing/SSP3D)
  * [Bilateral Normal Integration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545.pdf)<br>:star:[code](https://github.com/hoshino042/bilateral_normal_integration)
  * [CHORE: Contact, Human and Object REconstruction from a Single RGB Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620121.pdf)<br>:star:[code](https://github.com/xiexh20/CHORE):house:[project](https://virtualhumans.mpi-inf.mpg.de/chore/)
  * [Directed Ray Distance Functions for 3D Scene Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620193.pdf)<br>:house:[project](https://nileshkulkarni.github.io/scene_drdf)  
  * [Object Wake-Up: 3D Object Rigging from a Single Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620302.pdf)<br>:star:[code](https://drive.google.com/drive/folders/1tR-Q9Lna8-O2i2uUR4QAmGAZ6hHa1Llf):house:[project](https://kulbear.github.io/object-wakeup/)
  * [Latent Partition Implicit with Surface Codes for 3D Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630318.pdf)<br>:star:[code](https://github.com/chenchao15/LPI)
  * [3D Equivariant Graph Implicit Functions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630477.pdf)
  * [Projective Parallel Single-Pixel Imaging to Overcome Global Illumination in 3D Structure Light Scanning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660479.pdf)
  * [EvAC3D: From Event-Based Apparent Contours to 3D Models via Continuous Visual Hulls](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670278.pdf)<br>:house:[project](https://www.cis.upenn.edu/~ziyunw/evac3d/)
  * [3D CoMPaT: Composition of Materials on Parts of 3D Things](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680107.pdf)<br>:house:[project](https://3dcompat-dataset.org/)
* 三维形状
  * [Texturify: Generating Textures on 3D Shape Surfaces](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630073.pdf)
  * [Implicit Field Supervision for Robust Non-rigid Shape Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630338.pdf)<br>:star:[code](https://github.com/Sentient07/IFMatch)
  * [3D Shape Sequence of Human Comparison and Classification using Current and Varifolds](https://arxiv.org/abs/2207.12485)<br>:star:[code](https://github.com/CRISTAL-3DSAM/HumanComparisonVarifolds)
  * [The Shape Part Slot Machine: Contact-Based Reasoning for Generating 3D Shapes from Parts](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630599.pdf) 
 * 3D形状匹配
    * [Unsupervised Deep Multi-Shape Matching](https://arxiv.org/abs/2207.09610)
  * 3D形状合成
    * [Cross-Modal 3D Shape Generation and Manipulation](https://arxiv.org/abs/2207.11795)<br>:star:[code](https://github.com/snap-research/edit3d):house:[project](https://people.cs.umass.edu/~zezhoucheng/edit3d/)
  * 形状补全
    * [PatchRD: Detail-Preserving Shape Completion by Learning Patch Retrieval and Deformation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630494.pdf)<br>:star:[code](https://github.com/GitBoSun/PatchRD)
  * 形状解析
    * [ExtrudeNet: Unsupervised Inverse Sketch-and-Extrude for Shape Parsing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620468.pdf)<br>:star:[code](https://github.com/kimren227/ExtrudeNet) 
  * 形状修补
    * [DeepMend: Learning Occupancy Functions to Represent Shape for Repair](https://arxiv.org/abs/2210.05728)<br>:star:[code](https://github.com/Terascale-All-sensing-Research-Studio/DeepMend)
* depth restoration
  * [Domain Randomization-Enhanced Depth Simulation and Restoration for Perceiving and Grasping Specular and Transparent Objects](https://arxiv.org/abs/2208.03792)<br>:star:[code](https://github.com/PKU-EPIC/DREDS)   
* 场景理解
  * [Spatially Invariant Unsupervised 3D Object-Centric Learning and Scene Decomposition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830120.pdf)
  * [Pose2Room: Understanding 3D Scenes from Human Activities](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870418.pdf)
  * [Inverted Pyramid Multi-task Transformer for Dense Scene Understanding](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870506.pdf)<br>:star:[code](https://github.com/prismformore/InvPT)
  * [Panoptic-PartFormer: Learning a Unified Model for Panoptic Part Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870716.pdf)<br>:star:[code](https://github.com/lxtGH/Panoptic-PartFormer)
  * [Weakly Supervised 3D Scene Segmentation with Region-Level Boundary Awareness and Instance Discrimination](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880036.pdf)

<a name="7"/>

## 7.Object Tracking(目标跟踪)
* [Towards Grand Unification of Object Tracking](https://arxiv.org/abs/2207.07078)<br>:open_mouth:oral:star:[code](https://github.com/MasterBin-IIAU/Unicorn)<br>:newspaper:[ECCV 2022 Oral《Unicorn》首次统一了四项目标跟踪任务的网络结构与学习范式，在8个富有挑战性的数据集上SOTA](https://mp.weixin.qq.com/s/bB0g9MaC7I_x6hB_3fNcfQ)
* [HVC-Net: Unifying Homography, Visibility, and Confidence Learning for Planar Object Tracking](https://arxiv.org/abs/2209.08924)
* [Tracking by Associating Clips](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850126.pdf)
* [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)<br>:star:[code](https://github.com/ifzhang/ByteTrack)
* [Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820332.pdf)<br>:star:[code](https://github.com/botaoye/OSTrack)
* [Backbone Is All Your Need: A Simplified Architecture for Visual Object Tracking](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820366.pdf)<br>:star:[code](https://github.com/LPXTT/SimTrack)
* [Robust Visual Tracking by Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820555.pdf)<br>:star:[code](https://github.com/visionml/pytracking)
* [FEAR: Fast, Efficient, Accurate and Robust Visual Tracker](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820625.pdf)<br>:star:[code](https://github.com/PinataFarms/FEARTracker)
* 3D跟踪
  * [3D Siamese Transformer Network for Single Object Tracking on Point Clouds](https://arxiv.org/abs/2207.11995)<br>:star:[code](https://github.com/fpthink/STNet)
  * [Large-displacement 3D Object Tracking with Hybrid Non-local Optimization](https://arxiv.org/abs/2207.12620)<br>:star:[code](https://github.com/cvbubbles/nonlocal-3dtracking)
  * [CMT: Context-Matching-Guided Transformer for 3D Tracking in Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820091.pdf)
  * [Towards Generic 3D Tracking in RGBD Videos: Benchmark and Baseline](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820108.pdf)<br>:star:[code](https://github.com/yjybuaa/Track-it-in-3D)
* 多目标跟踪
  * [Tracking Objects as Pixel-wise Distributions](https://arxiv.org/abs/2207.05518)<br>:open_mouth:oral
  * [The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting](https://arxiv.org/abs/2207.09295)
  * [MOTCOM: The Multi-Object Tracking Dataset Complexity Metric](https://arxiv.org/abs/2207.10031)<br>:star:[code](https://github.com/JoakimHaurum/MOTCOM):house:[project](https://vap.aau.dk/motcom/)
  * [Tracking Every Thing in the Wild](https://arxiv.org/abs/2207.12978)
  * [PolarMOT: How Far Can Geometric Relations Take Us in 3D Multi-Object Tracking?](https://arxiv.org/abs/2208.01957)
  * [SOMPT22: A Surveillance Oriented Multi-Pedestrian Tracking Dataset](https://arxiv.org/abs/2208.02580)
  * [Robust Multi-Object Tracking by Marginal Inference](https://arxiv.org/abs/2208.03727)
  * [MOTR: End-to-End Multiple-Object Tracking with TRansformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870648.pdf)<br>:star:[code](https://github.com/megvii-research/MOTR)
  * [Large Scale Real-World Multi-person Tracking](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680493.pdf)<br>:star:[code](https://github.com/amazon-science/tracking-dataset)
  * [Particle Video Revisited: Tracking through Occlusions Using Point Trajectories](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820055.pdf)<br>:house:[project](https://particle-video-revisited.github.io/)
* 视觉跟踪
  * [AiATrack: Attention in Attention for Transformer Visual Tracking](https://arxiv.org/abs/2207.09603)<br>:star:[code](https://github.com/Little-Podi/AiATrack)
  * [Towards Sequence-Level Training for Visual Tracking](https://arxiv.org/abs/2208.05810)<br>:star:[code](https://github.com/byminji/SLTtrack)
  * [Hierarchical Feature Embedding for Visual Tracking](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820418.pdf)<br>:star:[code](https://github.com/zxgravity/CIA)
* 细胞跟踪
  * [Graph Neural Network for Cell Tracking in Microscopy Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810602.pdf)<br>:star:[code](https://github.com/talbenha/cell-tracker-gnn)

<a name="6"/>

## 6.Object Detection(目标检测)
* [Should All Proposals be Treated Equally in Object Detection?](https://arxiv.org/abs/2207.03520)<br>:star:[code](https://github.com/liyunsheng13/dpp)
* [HEAD: HEtero-Assists Distillation for Heterogeneous Object Detectors](https://arxiv.org/abs/2207.05345)<br>:star:[code](https://github.com/LutingWang/HEAD)
* [Adversarially-Aware Robust Object Detector](https://arxiv.org/abs/2207.06202)<br>:open_mouth:oral:star:[code](https://github.com/7eu7d7/RobustDet)
* [ObjectBox: From Centers to Boxes for Anchor-Free Object Detection](https://arxiv.org/abs/2207.06985)<br>:open_mouth:oral:star:[code](https://github.com/MohsenZand/ObjectBox)
* [Point-to-Box Network for Accurate Object Detection via Single Point Supervision](https://arxiv.org/abs/2207.06827)<br>:star:[code](https://github.com/ucas-vg/P2BNet)
* [You Should Look at All Objects](https://arxiv.org/abs/2207.07889)<br>:star:[code](https://github.com/CharlesPikachu/YSLAO)
* [Class-agnostic Object Detection with Multi-modal Transformer](https://arxiv.org/abs/2111.11430)<br>:star:[code](https://github.com/mmaaz60/mvits_for_class_agnostic_od)<br>使用多模态 ViTs 和人类可理解的文本查询来生成高质量的OP
* [Exploiting Unlabeled Data with Vision and Language Models for Object Detection](https://arxiv.org/abs/2207.08954)<br>:star:[code](https://github.com/xiaofeng94/VL-PLM) 
* [PoserNet: Refining Relative Camera Poses Exploiting Object Detections](https://arxiv.org/abs/2207.09445)<br>:star:[code](https://github.com/IIT-PAVIS/PoserNet)
* [Robust Object Detection With Inaccurate Bounding Boxes](https://arxiv.org/abs/2207.09697)<br>:star:[code](https://github.com/cxliu0/OA-MIL)
* [UC-OWOD: Unknown-Classified Open World Object Detection](https://arxiv.org/abs/2207.11455)<br>:star:[code](https://github.com/JohnWuzh/UC-OWOD)
* [Exploring Resolution and Degradation Clues as Self-supervised Signal for Low Quality Object](https://arxiv.org/abs/2208.03062)<br>:star:[code](https://github.com/cuiziteng/ECCV_AERIS)
* [Unifying Visual Perception by Dispersible Points Learning](https://arxiv.org/abs/2208.08630)<br>:star:[code](https://github.com/Sense-X/UniHead)
* [A Large-scale Multiple-objective Method for Black-box Attack against Object Detection](https://arxiv.org/abs/2209.07790)<br>:star:[code](https://github.com/LiangSiyuan21/GARSDC)
* [Distilling Object Detectors With Global Knowledge](https://arxiv.org/abs/2210.09022)<br>:star:[code](https://github.com/hikvision-research/DAVAR-Lab-ML)
* [PANDORA: A Panoramic Detection Dataset for Object with Orientation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680229.pdf)<br>:star:[code](https://github.com/tdsuper/SphericalObjectDetection)
* [Exploring Plain Vision Transformer Backbones for Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690276.pdf)<br>:star:[code](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
* [Long-Tail Detection with Effective Class-Margins](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680684.pdf)<br>:star:[code](https://github.com/janghyuncho/ECM-Loss)
* [Detecting Twenty-Thousand Classes Using Image-Level Supervision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690344.pdf)<br>:star:[code](https://github.com/facebookresearch/Detic)
* [Exploring Resolution and Degradation Clues As Self-Supervised Signal for Low Quality Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690465.pdf)<br>:star:[code](https://github.com/cuiziteng/ECCV_AERIS)
* [Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690501.pdf)
* [MTTrans: Cross-Domain Object Detection with Mean Teacher Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690620.pdf)
* [PromptDet: Towards Open-Vocabulary Detection Using Uncurated Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690691.pdf)<br>:house:[project](https://fcjian.github.io/promptdet)
* [Cornerformer: Purifying Instances for Corner-Based Detectors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700017.pdf)
* [Efficient Decoder-Free Object Detection with Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700069.pdf)<br>:star:[code](https://github.com/peixianchen/DFFT)
* [W2N: Switching from Weak Supervision to Noisy Supervision for Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900699.pdf)<br>:star:[code](https://github.com/1170300714/w2n_wsod)
* [Towards Data-Efficient Detection Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690090.pdf)<br>:star:[code](https://github.com/encounter1997/DE-DETRs)
* [Open-Vocabulary DETR with Conditional Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690107.pdf)<br>:star:[code](https://github.com/yuhangzang/OV-DETR)
* [Prediction-Guided Distillation for Dense Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690123.pdf)<br>:star:[code](https://github.com/ChenhongyiYang/PGD)
* [Multimodal Object Detection via Probabilistic Ensembling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)<br>:star:[code](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)
* [Open Vocabulary Object Detection with Pseudo Bounding-Box Labels](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700263.pdf)
* [GLAMD: Global and Local Attention Mask Distillation for Object Detectors](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700456.pdf)
* [Object Detection As Probabilistic Set Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700545.pdf)
* [Out-of-Distribution Identification: Let Detector Tell Which I Am Not Sure](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700631.pdf)
* [Simple Open-Vocabulary Object Detection with Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf)<br>:star:[code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
* [A Simple Approach and Benchmark for 21,000-Category Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710001.pdf)<br>:star:[code](https://github.com/SwinTransformer/Simple-21K-Detection)
* [EAutoDet: Efficient Architecture Search for Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800652.pdf)<br>:star:[code](https://github.com/vicFigure/EAutoDet) 
* [Few-Shot End-to-End Object Detection via Constantly Concentrated Encoding across Heads](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860056.pdf) 
* 3D目标检测
  * [DID-M3D: Decoupling Instance Depth for Monocular 3D Object Detection](https://arxiv.org/abs/2207.08531)<br>:star:[code](https://github.com/SPengLiang/DID-M3D)
  * [Rethinking IoU-based Optimization for Single-stage 3D Object Detection](https://arxiv.org/abs/2207.09332)<br>:star:[code](https://github.com/hlsheng1/RDIoU)
  * [Densely Constrained Depth Estimator for Monocular 3D Object Detection](https://arxiv.org/abs/2207.10047)<br>:star:[code](https://github.com/BraveGroup/DCD)
  * [Learning Ego 3D Representation As Ray Tracing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860126.pdf)<br>:house:[project](https://fudan-zvg.github.io/Ego3RT/)
  * [AutoAlignV2: Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection](https://arxiv.org/abs/2207.10316)<br>:star:[code](https://github.com/zehuichen123/AutoAlignV2)
  * [DEVIANT: Depth EquiVarIAnt NeTwork for Monocular 3D Object Detection](https://arxiv.org/abs/2207.10758)<br>:star:[code](https://github.com/abhi1kumar/DEVIANT)
  * [Label-Guided Auxiliary Training Improves 3D Object Detector](https://arxiv.org/abs/2207.11753)<br>:star:[code](https://github.com/FabienCode/LG3D)
  * [Monocular 3D Object Detection with Depth from Motion](https://arxiv.org/abs/2207.12988)<br>:open_mouth:oral:star:[code](https://github.com/Tai-Wang/Depth-from-Motion)
  * [MV-FCOS3D++: Multi-View Camera-Only 4D Object Detection with Pretrained Monocular Backbones](https://arxiv.org/abs/2207.12716)<br>:open_mouth:oral:star:[code](https://github.com/Tai-Wang/Depth-from-Motion)
  * [Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph](https://arxiv.org/abs/2208.03624)<br>:open_mouth:oral:star:[code](https://github.com/Nightmare-n/GraphRCNN) 
  * [CenterFormer: Center-based Transformer for 3D Object Detection](https://arxiv.org/abs/2209.05588)<br>:open_mouth:oral:star:[code](https://github.com/TuSimple/centerformer)
  * [SWFormer: Sparse Window Transformer for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2210.07372)
  * [Autoregressive Uncertainty Modeling for 3D Bounding Box Prediction](https://arxiv.org/abs/2210.07424)<br>:star:[code](https://github.com/naver/garnet)
  * [CramNet: Camera-Radar Fusion with Ray-Constrained Cross-Attention for Robust 3D Object Detection](https://arxiv.org/abs/2210.09267)
  * [Homogeneous Multi-modal Feature Fusion and Interaction for 3D Object Detection](https://arxiv.org/abs/2210.09615)
  * [Plausibility Verification For 3D Object Detectors Using Energy-Based Optimization](https://arxiv.org/abs/2211.05233)
  * [Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection](https://arxiv.org/abs/2211.07171)<br>:star:[code](https://github.com/Cc-Hy/CMKD)
  * [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870523.pdf)<br>:star:[code](https://github.com/megvii-research/PETR)
  * [Lidar Point Cloud Guided Monocular 3D Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610123.pdf)<br>:star:[code](https://github.com/SPengLiang/LPCG)  
  * [INT: Towards Infinite-Frames 3D Detection with an Efficient Framework](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690190.pdf)
  * [Semi-Supervised Monocular 3D Object Detection by Multi-View Consistency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680702.pdf)
  * [Unsupervised Domain Adaptation for Monocular 3D Object Detection via Self-Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690242.pdf)<br>:star:[code](https://github.com/zhyever/STMono3D)
  * [MPPNet: Multi-Frame Feature Intertwining with Proxy Points for 3D Temporal Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680667.pdf)<br>:star:[code](https://github.com/open-mmlab/OpenPCDet)
  * [PillarNet: Real-Time and High-Performance Pillar-Based 3D Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700034.pdf)<br>:star:[code](https://github.com/VISION-SJTU/PillarNet)
  * [Improving the Intra-Class Long-Tail in 3D Detection via Rare Example Mining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700155.pdf)
  * [3D Object Detection with a Self-Supervised Lidar Scene Flow Backbone](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700244.pdf)<br>:star:[code](https://github.com/emecercelik/ssl-3d-detection)
  * [DetMatch: Two Teachers Are Better than One for Joint 2D and 3D Semi-Supervised Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700366.pdf)<br>:star:[code](https://github.com/Divadi/DetMatch)
  * [FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700473.pdf)<br>:star:[code](https://github.com/samsunglabs/fcaf3d)
  * [Enhancing Multi-modal Features Using Local Self-Attention for 3D Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700527.pdf)
* 半监督目标检测
  * [Dense Teacher: Dense Pseudo-Labels for Semi-supervised Object Detection](https://arxiv.org/abs/2207.02541)<br>:star:[code](https://github.com/Megvii-BaseDetection/DenseTeacher)
  * [Open-Set Semi-Supervised Object Detection](https://arxiv.org/abs/2208.13722)<br>:star:[code](https://github.com/facebookresearch/OSSOD):house:[project](https://ycliu93.github.io/projects/ossod.html)
  * [PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690449.pdf)<br>:star:[code](https://github.com/ligang-cs/PseCo)
  * [Diverse Learner: Exploring Diverse Supervision for Semi-Supervised Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900631.pdf)
* 小样本目标检测
  * [Rethinking Few-Shot Object Detection on a Multi-Domain Benchmark](https://arxiv.org/abs/2207.11169)<br>:star:[code](https://github.com/amazon-research/few-shot-object-detection-benchmark)
  * [Multi-Faceted Distillation of Base-Novel Commonality for Few-shot Object Detection](https://arxiv.org/abs/2207.11184)<br>:star:[code](https://github.com/WuShuang1998/MFDC)
  * [AcroFOD: An Adaptive Method for Cross-domain Few-shot Object Detection](https://arxiv.org/abs/2209.10904)<br>:star:[code](https://github.com/Hlings/AcroFOD)
  * [Time-rEversed diffusioN tEnsor Transformer: A new TENET of Few-Shot Object Detection](https://arxiv.org/abs/2210.16897)<br>:star:[code](https://github.com/ZS123-lang/TENET)
  * [Few-Shot Object Detection by Knowledge Distillation Using Bag-of-Visual-Words Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700279.pdf)
   * [Few-Shot Object Detection with Model Calibration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790707.pdf)<br>:star:[code](https://github.com/fanq15/FewX)
  * [Few-Shot Video Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800071.pdf)<br>:star:[code](https://github.com/fanq15/FewX)
  * [Mutually Reinforcing Structure with Proposal Contrastive Consistency for Few-Shot Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800388.pdf)<br>:star:[code](https://github.com/MMatx/MRSN)
* 显著目标检测  
  * [SESS: Saliency Enhancing with Scaling and Sliding](https://arxiv.org/abs/2207.01769)<br>:star:[code](https://github.com/neouyghur/SESS)
  * [SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection](https://arxiv.org/abs/2207.07898)<br>:star:[code](https://github.com/Hydragon516/SPSN)
  * [Salient Object Detection for Point Clouds](https://arxiv.org/abs/2207.11889)<br>:star:[code](https://git.openi.org.cn/OpenPointCloud/PCSOD)
  * [KD-SCFNet: Towards More Accurate and Efficient Salient Object Detection via Knowledge Distillation](https://arxiv.org/abs/2208.02178)<br>:star:[code](https://github.com/zhangjinCV/KD-SCFNet)
  * [Saliency Hierarchy Modeling via Generative Kernels for Salient Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880564.pdf)
  * [MVSalNet:Multi-View Augmentation for RGB-D Salient Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890268.pdf) 
* 弱监督目标检测
  * [Active Learning Strategies for Weakly-supervised Object Detection](https://arxiv.org/abs/2207.12112)<br>:star:[code](https://github.com/huyvvo/BiB)
  * [W2N:Switching From Weak Supervision to Noisy Supervision for Object Detection](https://arxiv.org/abs/2207.12104)<br>:star:[code](https://github.com/1170300714/w2n_wsod)
  * [Object Discovery via Contrastive Learning for Weakly Supervised Object Detection](https://arxiv.org/abs/2208.07576)<br>:star:[code](https://github.com/jinhseo/OD-WSCL)
  * [End-to-End Weakly Supervised Object Detection with Sparse Proposal Evolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690207.pdf)
* 目标定位
  * [On Label Granularity and Object Localization](https://arxiv.org/abs/2207.10225)
  * 弱监督目标定位
    * [Bagging Regional Classification Activation Maps for Weakly Supervised Object Localization](https://arxiv.org/abs/2207.07818)<br>:star:[code](https://github.com/zh460045050/BagCAMs)
    * [Weakly Supervised Object Localization through Inter-class Feature Similarity and Intra-Class Appearance Consistency](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900194.pdf)
    * [Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration](https://arxiv.org/abs/2207.10447)<br>:star:[code](https://github.com/164140757/SCM)
* 单阶目标检测
  * [Unsupervised Domain Adaptation for One-stage Object Detector using Offsets to Bounding Box](https://arxiv.org/abs/2207.09656)
* 目标计数
  * [Few-shot Object Counting and Detection](https://arxiv.org/abs/2207.10988)<br>:star:[code](https://github.com/VinAIResearch/Counting-DETR)
* OOD
  * [Out-of-Distribution Detection with Semantic Mismatch under Masking](https://arxiv.org/abs/2208.00446)<br>:star:[code](https://github.com/cure-lab/MOODCat)
  * [Out-of-Distribution Detection with Boundary Aware Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840232.pdf) 
  * [DICE: Leveraging Sparsification for Out-of-Distribution Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840680.pdf)<br>:star:[code](https://github.com/deeplearning-wisc/dice)
  * [Class Is Invariant to Context and Vice Versa: On Learning Invariance for Out-of-Distribution Generalization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850089.pdf)<br>:star:[code](https://github.com/simpleshinobu/IRMCon)
* VOD
  * [PTSEFormer: Progressive Temporal-Spatial Enhanced TransFormer towards Video Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680719.pdf)<br>:star:[code](https://github.com/Hon-Wong/PTSEFormer)
  * [SALISA: Saliency-Based Input Sampling for Efficient Video Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700296.pdf) 
  * [Bridging Images and Videos: A Simple Learning Framework for Large Vocabulary Video Object Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850235.pdf)
* 小目标检测
  * [RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection](https://arxiv.org/abs/2208.08738)<br>:star:[code](https://github.com/Chasel-Tsui/mmdet-rfla) 
* 图像检测
  * [Discovering Transferable Forensic Features for CNN-generated Images Detection](https://arxiv.org/abs/2208.11342)<br>:open_mouth:oral:star:[code](https://drive.google.com/drive/folders/1LvKIwHf5dEbm-MxvAMQRiGViWLSKYfRP?usp=sharing):house:[project](https://keshik6.github.io/transferable-forensic-features/)
* 目标发现
  * [Object Discovery and Representation Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870121.pdf)

<a name="5"/>

## 5.Image/Video Retrieval(图像/视频检索)
* [Text-Based Temporal Localization of Novel Events](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740552.pdf)
* 跨域检索
  * [Feature Representation Learning for Unsupervised Cross-domain Image Retrieval](https://arxiv.org/abs/2207.09721)<br>:star:[code](https://github.com/conghuihu/UCDIR)
* 图像检索
  * [Hierarchical Average Precision Training for Pertinent Image Retrieval](https://arxiv.org/abs/2207.04873)<br>:star:[code](https://github.com/elias-ramzi/HAPPIER)
  * [Adaptive Fine-Grained Sketch-Based Image Retrieval](https://arxiv.org/abs/2207.01723)<br>:star:[code](https://github.com/AyanKumarBhunia/Adaptive-FGSBIR)
  * [A Sketch Is Worth a Thousand Words: Image Retrieval with Text and Sketch](https://arxiv.org/abs/2208.03354)<br>:star:[code](https://github.com/janesjanes/tsbir):house:[project](https://patsorn.me/projects/tsbir/)
  * [Granularity-aware Adaptation for Image Retrieval over Multiple Tasks](https://arxiv.org/abs/2210.02254)
  * [Reliability-Aware Prediction via Uncertainty Learning for Person Image Retrieval](https://arxiv.org/abs/2210.13440)<br>:star:[code](https://github.com/dcp15/UAL)
  * [StyleBabel: Artistic Style Tagging and Captioning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680212.pdf)
  * [Deep Hash Distillation for Image Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740345.pdf)<br>:star:[code](https://github.com/youngkyunJang/Deep-Hash-Distillation)
  * [Conditional Stroke Recovery for Fine-Grained Sketch-Based Image Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860708.pdf)
  * [Fine-Grained Fashion Representation Learning by Online Deep Clustering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870019.pdf)
* 视频检索
  * [LocVTP: Video-Text Pre-training for Temporal Localization](https://arxiv.org/abs/2207.10362)<br>:star:[code](https://github.com/mengcaopku/LocVTP)
  * [Dual-Stream Knowledge-Preserving Hashing for Unsupervised Video Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740175.pdf)
  * [Multi-Query Video Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740227.pdf)<br>:star:[code](https://github.com/princetonvisualai/MQVR)
  * [Learning Audio-Video Modalities from Image Captions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740396.pdf)<br>:house:[project](https://a-nagrani.github.io/videocc.html)
  * [Audio-Visual Mismatch-Aware Video Retrieval via Association and Adjustment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740484.pdf)
  * Video Geo-localization(检索)
    * [GAMa: Cross-view Video Geo-localization](https://arxiv.org/abs/2207.02431)<br>:star:[code](https://github.com/svyas23/GAMa)
* 文本-视频检索
  * [TS2-Net: Token Shift and Selection Transformer for Text-Video Retrieval](https://arxiv.org/abs/2207.07852)<br>:star:[code](https://github.com/yuqi657/ts2_net)
  * [Lightweight Attentional Feature Fusion: A New Baseline for Text-to-Video Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740432.pdf)<br>:star:[code](https://github.com/ruc-aimc-lab/laff)
* 图像-文本检索
  * [CODER: Coupled Diversity-Sensitive Momentum Contrastive Learning for Image-Text Retrieval](https://arxiv.org/abs/2208.09843)
* 细粒度图像检索
  * [SEMICON: A Learning-to-hash Solution for Large-scale Fine-grained Image Retrieval](https://arxiv.org/abs/2209.13833)<br>:star:[code](https://github.com/NJUST-VIPGroup/SEMICON)
* 视频时刻检索
  * [Selective Query-guided Debiasing Network for Video Corpus Moment Retrieval](https://arxiv.org/abs/2210.08714)
* 视频-文本检索
  * [VTC: Improving Video-Text Retrieval with User Comments](https://arxiv.org/abs/2210.10820)<br>:star:[code](https://github.com/unitaryai/VTC):house:[project](https://unitaryai.github.io/vtc-paper/)
* 最近邻搜索
  * [Connecting Compression Spaces with Transformer for Approximate Nearest Neighbor Search](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740502.pdf)<br>:star:[code](https://github.com/hkzhang91/CCST)

<a name="4"/>

## 4.Video/Image Captioning(视频/图像字幕)
* 图像字幕
  * [GRIT: Faster and Better Image captioning Transformer Using Dual Visual Features](https://arxiv.org/abs/2207.09666)<br>:star:[code](https://github.com/davidnvq/grit)
  * [Explicit Image Caption Editing](https://arxiv.org/abs/2207.09625)<br>:star:[code](https://github.com/baaaad/ECE)
  * [ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-Verified Image-Caption Associations for MS-COCO](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680001.pdf)<br>:star:[code](https://github.com/naver-ai/eccv-caption)

<a name="3"/>

## 3.Image Progress(图像处理)
* 图像质量评估
  * [Shift-tolerant Perceptual Similarity Metric](https://arxiv.org/abs/2207.13686)<br>:star:[code](https://github.com/abhijay9/ShiftTolerant-LPIPS/)
* 图像修补(image retouching)
  * [Neural Color Operators for Sequential Image Retouching](https://arxiv.org/abs/2207.08080)<br>:star:[code](https://github.com/amberwangyili/neurop)
* 图像变形(Image Warping)
  * [Learning Local Implicit Fourier Representation for Image Warping](https://arxiv.org/abs/2207.01831)<br>:star:[code](https://github.com/jaewon-lee-b/ltew)
* 图像恢复
  * [D2HNet: Joint Denoising and Deblurring with Hierarchical Network for Robust Night Image Restoration](https://arxiv.org/abs/2207.03294)<br>:star:[code](https://github.com/zhaoyuzhi/D2HNet)
  * [Simple Baselines for Image Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)<br>:star:[code](https://github.com/megvii-research/NAFNet)
  * [Improving Image Restoration by Revisiting Global Information Aggregation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670053.pdf)<br>:star:[code](https://github.com/megvii-research/tlc)
  * [Seeing through a Black Box: Toward High-Quality Terahertz Imaging via Subspace-and-Attention Guided Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670447.pdf)
  * [JPEG Artifacts Removal via Contrastive Representation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770618.pdf)<br>:star:[code](https://github.com/wang-xi-1/JPEG)
  * [TAPE: Task-Agnostic Prior Embedding for Image Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780438.pdf)
  * [Spectrum-Aware and Transferable Architecture Search for Hyperspectral Image Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790019.pdf)
  * [DRCNet: Dynamic Image Restoration Contrastive Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790504.pdf)
* 图像修复
  * [Learning Prior Feature and Attention Enhanced Image Inpainting](https://arxiv.org/abs/2208.01837)<br>:star:[code](https://github.com/ewrfcas/MAE-FAR)
  * [Inpainting at Modern Camera Resolution by Guided PatchMatch with Auto-Curation](https://arxiv.org/abs/2208.03552)<br>:star:[code](https://drive.google.com/file/d/1Lmar1byASRReJ0SimBfWAE2Dt_-gMhor/edit)
  * [High-Fidelity Image Inpainting with GAN Inversion](https://arxiv.org/abs/2208.11850)
  * [Unbiased Multi-Modality Guidance for Image Inpainting](https://arxiv.org/abs/2208.11844)
  * [Image Inpainting with Cascaded Modulation GAN and Object-Aware Training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760263.pdf)<br>:star:[code](https://github.com/htzheng/CM-GAN-Inpainting)
  * [Perceptual Artifacts Localization for Inpainting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890145.pdf)<br>:star:[code](https://github.com/owenzlz/PAL4Inpaint)
  * [Hourglass Attention Network for Image Inpainting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780474.pdf)<br>:star:[code](https://github.com/dengyecode/hourglassattention) 
  * [Diverse Image Inpainting with Normalizing Flow](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830053.pdf)
* 图像增强
  * [SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement](https://arxiv.org/abs/2207.08351)
  * [Uncertainty Inspired Underwater Image Enhancement](https://arxiv.org/abs/2207.09689)<br>:star:[code](https://github.com/zhenqifu/PUIE-Net)
  * [Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression](https://arxiv.org/abs/2207.10564)<br>:star:[code](https://github.com/jinyeying/night-enhancement)
  * [LEDNet: Joint Low-Light Enhancement and Deblurring in the Dark](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660562.pdf)<br>:star:[code](https://github.com/sczhou/LEDNet):house:[project](https://shangchenzhou.com/projects/LEDNet/)
  * [NEST: Neural Event Stack for Event-Based Image Enhancement](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660649.pdf)<br>:star:[code](https://github.com/ChipsAhoyM/NEST)
  * [Seeing Far in the Dark with Patterned Flash](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660698.pdf)<br>:star:[code](https://github.com/zhsun0357/Seeing-Far-in-the-Dark-with-Patterned-Flash)
  * [Local Color Distributions Prior for Image Enhancement](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780336.pdf)<br>:house:[project](https://hywang99.github.io/lcdpnet/)
* 图像和谐化 
  * [DCCF: Deep Comprehensible Color Filter Learning Framework for High-Resolution Image Harmonization](https://arxiv.org/abs/2207.04788)<br>:open_mouth:oral:star:[code](https://github.com/rockeyben/DCCF)
  * [Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670327.pdf)<br>:star:[code](http://github.com/stefanLeong/S2CRNet)
* 图像去卷积
  * [Learning Discriminative Shrinkage Deep Networks for Image Deconvolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790212.pdf)
* 去雾
  * [Boosting Supervised Dehazing Methods via Bi-Level Patch Reweighting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780055.pdf)
  * [Unpaired Deep Image Dehazing Using Contrastive Disentanglement Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770636.pdf)
  * [Perceiving and Modeling Density for Image Dehazing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790126.pdf)<br>:star:[code](https://github.com/Owen718/ECCV22-Perceiving-and-Modeling-Density-for-Image-Dehazing)
  * [Frequency and Spatial Dual Guidance for Image Dehazing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790177.pdf)<br>:star:[code](https://github.com/yuhuUSTC/FSDGN)
* 去噪
  * [Deep Semantic Statistics Matching (D2SM) Denoising Network](https://arxiv.org/abs/2207.09302)<br>:star:[code](https://github.com/MKFMIKU/d2sm):house:[project](https://kfmei.page/d2sm/)
  * [Optimizing Image Compression via Joint Learning with Denoising](https://arxiv.org/abs/2207.10869)<br>:star:[code](https://github.com/felixcheng97/DenoiseCompression)
  * [Fast and High Quality Image Denoising via Malleable Convolution](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780420.pdf)<br>:house:[project](https://yifanjiang.net/MalleConv.html)
  * [Unidirectional Video Denoising by Mimicking Backward Recurrent Modules with Look-Ahead Forward Ones](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780581.pdf)<br>:star:[code](https://github.com/nagejacob/FloRNN)
  * [TempFormer: Temporally Consistent Transformer for Video Denoising](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790471.pdf)
* 去雪
  * [SLiDE: Self-supervised LiDAR De-snowing through Reconstruction Difficulty](https://arxiv.org/abs/2208.04043) 
* 去雨
  * [Not Just Streaks: Towards Ground Truth for Single Image Deraining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670713.pdf)<br>:house:[project](https://visual.ee.ucla.edu/gt_rain.htm/)
  * [Blind Image Decomposition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780214.pdf)<br>:star:[code](https://github.com/JunlinHan/BID)
  * [ART-SS: An Adaptive Rejection Technique for Semi-Supervised Restoration for Adverse Weather-Affected Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780688.pdf)<br>:star:[code](https://github.com/rajeevyasarla/ART-SS)
  * [Rethinking Video Rain Streak Removal: A New Synthesis Model and a Deraining Network with Video Rain Prior](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790556.pdf)<br>:star:[code](https://github.com/wangshauitj/RDD-Net)
* 去模糊
  * [Animation from Blur: Multi-modal Blur Decomposition with Motion Guidance](https://arxiv.org/abs/2207.10123)<br>:star:[code](https://github.com/zzh-tech/Animation-from-Blur)
  * [United Defocus Blur Detection and Deblurring via Adversarial Promoting Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900562.pdf)<br>:star:[code](https://github.com/wdzhao123/APL)
  * [Learning Degradation Representations for Image Deblurring](https://arxiv.org/abs/2208.05244)<br>:star:[code](https://github.com/dasongli1/Learning_degradation)
  * [Learning Deep Non-Blind Image Deconvolution without Ground Truths](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660631.pdf)
  * [DeMFI: Deep Joint Deblurring and Multi-Frame Interpolation with Flow-Guided Attentive Correlation and Recursive Boosting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670193.pdf)<br>:star:[code](https://github.com/JihyongOh/DeMFI)
  * [Realistic Blur Synthesis for Learning Image Deblurring](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670481.pdf)<br>:house:[project](http://cg.postech.ac.kr/research/RSBlur/)
  * [Stripformer: Strip Transformer for Fast Image Deblurring](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790142.pdf)<br>:star:[code](https://github.com/pp00704831/Stripformer)
  * [Event-Based Fusion for Motion Deblurring with Cross-Modal Attention](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780403.pdf)<br>:house:[project](https://ahupujr.github.io/EFNet/)
  * [ERDN: Equivalent Receptive Field Deformable Network for Video Deblurring](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780651.pdf)<br>:star:[code](https://github.com/TencentCloud/ERDN)
  * [Event-Guided Deblurring of Unknown Exposure Time Videos](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780510.pdf)<br>:house:[project](https://intelpro.github.io/UEVD/)
* 去摩尔纹
  * [Towards Efficient and Scale-Robust Ultra-High-Definition Image Demoireing](https://arxiv.org/abs/2207.09935)<br>:star:[code](https://github.com/CVMI-Lab/UHDM):house:[project](https://xinyu-andy.github.io/uhdm-page/)
* 去反射
  * [Zero-Shot Learning for Reflection Removal of Single 360-Degree Image](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790523.pdf)
* 去阴影
  * [Style-Guided Shadow Removal](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790353.pdf)<br>:star:[code](https://github.com/jinwan1994/SG-ShadowNet)
* 语义图像编辑
  * [Context-Consistent Semantic Image Editing with Style-Preserved Modulation](https://arxiv.org/abs/2207.06252)<br>:star:[code](https://github.com/WuyangLuo/SPMPGAN)
* 图像着色
  * [PalGAN: Image Colorization with Palette Generative Adversarial Networks](https://arxiv.org/abs/2210.11204)<br>:star:[code](https://github.com/shepnerd/PalGAN)
  * [Semantic-Sparse Colorization Network for Deep Exemplar-Based Colorization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660495.pdf)
  * [CT2: Colorization Transformer via Color Tokens](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670001.pdf)
  * [BigColor: Colorization Using a Generative Color Prior for Natural Images](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670343.pdf)
  * [ColorFormer: Image Colorization via Color Memory Assisted Hybrid-Attention Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760020.pdf)<br>:star:[code](https://github.com/marsggbo/EAGAN)
  * [Bridging the Domain Gap towards Generalization in Automatic Colorization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770530.pdf)<br>:star:[code](https://github.com/Lhyejin/DG-Colorization)
  * [L-CoDer: Language-Based Colorization with Color-Object Decoupling Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780352.pdf)
* 图像裁剪
  * [Human-Centric Image Cropping with Partition-Aware and Content-Preserving Features](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670176.pdf)<br>:star:[code](https://github.com/bcmi/Human-Centric-Image-Cropping)
* 图像融合
  * [Neural Image Representations for Multi-Image Fusion and Layer Separation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670210.pdf)<br>:house:[project](https://shnnam.github.io/research/nir/)
* Rolling shutter(果冻效应)
  * [Bringing Rolling Shutter Images Alive with Dual Reversed Distortion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670227.pdf)<br>:star:[code](https://github.com/zzh-tech/Dual-Reversed-RS)


<a name="2"/>

## 2.Image Segmentation(图像分割)
* [PseudoClick: Interactive Image Segmentation with Click Imitation](https://arxiv.org/abs/2207.05282)
* [GitNet: Geometric Prior-Based Transformation for Birds-Eye-View Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610390.pdf)
* [Highly Accurate Dichotomous Image Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780036.pdf)<br>:house:[project](https://xuebinqin.github.io/dis/index.html)
* [Graph-Constrained Contrastive Regularization for Semi-Weakly Volumetric Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810396.pdf)
* [Slim Scissors: Segmenting Thin Object from Synthetic Background](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890375.pdf)<br>:star:[code](https://kunyanghan.github.io/SlimScissors/)
* [RankSeg: Adaptive Pixel Classification with Image Category Ranking for Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890673.pdf)<br>:star:[code](https://github.com/openseg-group/RankSeg)
* [Unsupervised Segmentation in Real-World Images via Spelke Object Inference](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890708.pdf)
* 语义分割
  * [Multi-Exit Semantic Segmentation Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810326.pdf)
  * [Learning Implicit Feature Alignment Function for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890479.pdf)<br>:star:[code](https://github.com/hzhupku/IFA)
  * [Data Efficient 3D Learner via Knowledge Transferred from 2D Model](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890181.pdf)<br>:star:[code](https://github.com/bryanyu1997/Data-Efficient-3D-Learner)
  * [Multi-Scale and Cross-Scale Contrastive Learning for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890408.pdf)<br>:star:[code](https://github.com/RViMLab/ECCV2022-multi-scale-and-cross-scale-contrastive-segmentation)
  * [2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds](https://arxiv.org/abs/2207.04397)<br>:star:[code](https://github.com/yanx27/2DPASS)
  * [Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding](https://arxiv.org/abs/2207.08455)
  * [ML-BPM: Multi-teacher Learning with Bidirectional Photometric Mixing for Open Compound Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/2207.09045)
  * [Union-Set Multi-source Model Adaptation for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890570.pdf)
  * [Continual Semantic Segmentation via Structure Preserving and Projected Feature Alignment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890341.pdf)
  * [Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.10169)<br>:star:[code](https://github.com/JayQine/MGD-SSSS)
  * [LiDAL: Inter-Frame Uncertainty Based Active Learning for 3D LiDAR Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870245.pdf)<br>:star:[code](https://github.com/hzykent/LiDAL)
  * [DODA: Data-Oriented Sim-to-Real Domain Adaptation for 3D Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870280.pdf)<br>:star:[code](https://github.com/CVMI-Lab/DODA)
  * [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870592.pdf)<br>:star:[code](https://github.com/QingyongHu/SQN)
  * [Learning Semantic Segmentation from Multiple Datasets with Label Shifts](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880019.pdf)<br>:house:[project](https://www.nec-labs.com/~mas/UniSeg/)
  * [CAR: Class-Aware Regularizations for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880514.pdf)<br>:star:[code](https://github.com/edwardyehuang/CAR)
  * [Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880530.pdf)<br>:star:[code](https://github.com/HeliosZhao/SHADE)
  * [A Transformer-Based Decoder for Semantic Segmentation with Multi-level Context Mining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880617.pdf)<br>:star:[code](https://github.com/lygsbw/segdeformer)
  * [Extract Free Dense Labels from CLIP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880687.pdf)
  * [A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-Language Model](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890725.pdf)<br>:star:[code](https://github.com/MendelXu/zsseg.baseline)
  * [UCTNet: Uncertainty-Aware Cross-Modal Transformer Network for Indoor RGB-D Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900020.pdf)
  * [CP2: Copy-Paste Contrastive Pretraining for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900494.pdf)<br>:star:[code](https://github.com/wangf3014/CP2)
  * 域适应语义分割
    * [DecoupleNet: Decoupled Network for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2207.09988)<br>:star:[code](https://github.com/dvlab-research/DecoupleNet)
    * [HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900370.pdf)<br>:star:[code](https://github.com/lhoyer/HRDA)
    * [D2ADA: Dynamic Density-Aware Active Domain Adaptation for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890443.pdf)
    * [Online Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions](https://arxiv.org/abs/2207.10667)<br>:star:[code](https://github.com/theo2021/OnDA)
    * [Bi-directional Contrastive Learning for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2207.10892)<br>:star:[code](https://github.com/cvlab-yonsei/DASS):house:[project](https://cvlab.yonsei.ac.kr/projects/DASS/)
  * 小样本语义分割
    * [Self-Support Few-Shot Semantic Segmentation](https://arxiv.org/abs/2207.11549)<br>:star:[code](https://github.com/fanq15/SSP)
    * [Cross-Domain Few-Shot Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900072.pdf)<br>:star:[code](https://github.com/slei109/PATNet)
  * 弱监督语义分割
    * [Max Pooling with Vision Transformers reconciles class and shape in weakly supervised semantic segmentation](https://arxiv.org/abs/2210.17400)<br>:star:[code](https://github.com/deepplants/ViT-PCM)
    * [Adaptive Spatial-BCE Loss for Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890198.pdf)<br>:star:[code](https://github.com/allenwu97/Spatial-BCE)
  * [Adversarial Erasing Framework via Triplet with Gated Pyramid Pooling Layer for Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf)<br>:star:[code](https://github.com/KAIST-vilab/AEFT)
  * 无监督语义分割
    * [TransFGU: A Top-down Approach to Fine-Grained Unsupervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890072.pdf)<br>:star:[code](https://github.com/damo-cv/TransFGU)
* 实例分割 
  * [OSFormer: One-Stage Camouflaged Instance Segmentation with Transformers](https://arxiv.org/abs/2207.02255)<br>:star:[code](https://github.com/PJLallen/OSFormer)
  * [Geodesic-Former: A Geodesic-Guided Few-Shot 3D Point Cloud Instance Segmenter](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890552.pdf)<br>:star:[code](https://github.com/VinAIResearch/GeoFormer)
  * [Learning Regional Purity for Instance Segmentation on 3D Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900055.pdf)
  * [3D Instances as 1D Kernels](https://arxiv.org/abs/2207.07372)<br>:star:[code](https://github.com/W1zheng/DKNet)
  * [2D Amodal Instance Segmentation Guided by 3D Shape Prior](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890164.pdf)
  * [Box-supervised Instance Segmentation with Level Set Evolution](https://arxiv.org/abs/2207.09055)<br>:star:[code](https://github.com/LiWentomng/boxlevelset)
  * [Long-tailed Instance Segmentation using Gumbel Optimized Loss](https://arxiv.org/abs/2207.10936)<br>:star:[code](https://github.com/kostas1515/GOL)
  * [Active Pointly-Supervised Instance Segmentation](https://arxiv.org/abs/2207.11493)
  * [Trapped in Texture Bias? A Large Scale Comparison of Deep Instance Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680597.pdf)<br>:star:[code](https://github.com/JohannesTheo/trapped-in-texture-bias)
  * [Learning with Free Object Segments for Long-Tailed Instance Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700648.pdf)<br>:star:[code](https://github.com/czhang0528/FreeSeg)
  * [A Simple Single-Scale Vision Transformer for Object Detection and Instance Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700697.pdf)<br>:star:[code](https://github.com/tensorflow/models/tree/master/official/projects/uvit)
  * [Learning to Detect Every Thing in an Open World](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840265.pdf)<br>:house:[project](https://ksaito-ut.github.io/openworld_ldet/)  
* 全景分割
  * [Pointly-Supervised Panoptic Segmentation](https://arxiv.org/abs/2210.13950)<br>:star:[code](https://github.com/BraveGroup/PSPS)
* 运动分割
  * [Fast Two-View Motion Segmentation Using Christoffel Polynomials](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900001.pdf)
  * [Quantum Motion Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890497.pdf)
* 小样本分割
  * [Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation](https://arxiv.org/abs/2207.08549)<br>:star:[code](https://github.com/pawn-sxy/DCAMA)
  * [Cost Aggregation with 4D Convolutional Swin Transformer for Few-Shot Segmentation](https://arxiv.org/abs/2207.10866)<br>:star:[code](https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer):house:[project](https://seokju-cho.github.io/VAT/)
  * [Doubly Deformable Aggregation of Covariance Matrices for Few-shot Segmentation](https://arxiv.org/abs/2208.00306)<br>:star:[code](https://github.com/ShadowXZT/DACM-Few-shot.pytorch)
  * [Interclass Prototype Relation for Few-Shot Segmentation](https://arxiv.org/abs/2211.08681)
  * [HM: Hybrid Masking for Few-Shot Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800492.pdf)<br>:star:[code](https://github.com/moonsh/HM-Hybrid-Masking) 
  * [Adaptive Agent Transformer for Few-Shot Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890035.pdf)
  * [Dense Gaussian Processes for Few-Shot Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890215.pdf)<br>:star:[code](https://github.com/joakimjohnander/dgpnet)
* 抠图
  * [TransMatting: Enhancing Transparent Objects Matting with Transformers](https://arxiv.org/abs/2208.03007)<br>:star:[code](https://github.com/AceCHQ/TransMatting)
* 3D分割
  * [MvDeCor: Multi-view Dense Correspondence Learning for Fine-grained 3D Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620538.pdf)<br>:star:[code](https://github.com/nv-tlabs/MvDeCor):house:[project](https://nv-tlabs.github.io/MvDeCor/)
  * [PointInst3D: Segmenting 3D Instances by Points](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630284.pdf)
* 手分割
  * [Generative Adversarial Network for Future Hand Segmentation from Egocentric Video](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730638.pdf)<br>:house:[project](https://vjwq.github.io/EgoGAN/)
* 零件分割
 * [Improving Few-Shot Part Segmentation Using Coarse Supervision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900282.pdf) 
   
<a name="1"/>

## 1.其它
* [MVP: Multimodality-Guided Visual Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900336.pdf)
* [Self-Filtering: A Noise-Aware Sample Selection for Label Noise with Confidence Penalization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900511.pdf)
* [Learning to Learn with Smooth Regularization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830533.pdf)<br>:star:[code](https://github.com/xyh97/SmoothedOptimizer)
* [Ensemble Learning Priors Driven Deep Unfolding for Scalable Video Snapshot Compressive Imaging](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830583.pdf)<br>:star:[code](https://github.com/integritynoble/ELP-Unfolding/tree/master)
* [Approximate Discrete Optimal Transport Plan with Auxiliary Measure Method](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830602.pdf)
* [A Comparative Study of Graph Matching Algorithms in Computer Vision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830618.pdf)
* [Semidefinite Relaxations of Truncated Least-Squares in Robust Rotation Search: Tight or Not](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830655.pdf)
* [Dynamic Metric Learning with Cross-Level Concept Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840194.pdf)<br>:star:[code](https://github.com/wzzheng/CLCD)
* [MENet: A Memory-Based Network with Dual-Branch for Efficient Event Stream Processing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840211.pdf)
* [Improving Robustness by Enhancing Weak Subnets](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840317.pdf)
* [Learning from Multiple Annotator Noisy Labels via Sample-Wise Label Fusion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840402.pdf)<br>:star:[code](https://github.com/zhengqigao/Learning-from-Multiple-Annotator-Noisy-Labels)
* [Unbiased Manifold Augmentation for Coarse Class Subdivision](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850478.pdf)<br>:star:[code](https://github.com/leo-gb/UMA)
* [OccamNets: Mitigating Dataset Bias by Favoring Simpler Hypotheses](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800685.pdf)<br>:star:[code](https://github.com/erobic/occam-nets-v1)
* [ERA: Enhanced Rational Activations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800705.pdf)<br>:star:[code](https://github.com/martrim/ERA)
* [Active Label Correction Using Robust Parameter Update and Entropy Propagation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810001.pdf)
* [Revisiting Batch Norm Initialization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810207.pdf)<br>:star:[code](https://github.com/osu-cvl/revisiting-bn-init)
* [Differentiable Rendering for Synthetic Aperture Radar Imagery](https://arxiv.org/abs/2204.01248)
* [Batch-efficient EigenDecomposition for Small and Medium Matrices](https://arxiv.org/abs/2207.04228)<br>:star:[code](https://github.com/KingJamesSong/BatchED)
* [Accelerating Score-based Generative Models with Preconditioned Diffusion Sampling](https://arxiv.org/abs/2207.02196)
* [Improving Covariance Conditioning of the SVD Meta-layer by Orthogonality](https://arxiv.org/abs/2207.02119)<br>:star:[code](https://github.com/KingJamesSong/OrthoImproveCond)
* [Contrastive Deep Supervision](https://arxiv.org/abs/2207.05306)<br>:star:[code](https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision)
* [Organic Priors in Non-Rigid Structure from Motion](https://arxiv.org/abs/2207.06262)<br>:open_mouth:oral
* [Bootstrapped Masked Autoencoders for Vision BERT Pretraining](https://arxiv.org/abs/2207.07116)<br>:star:[code](https://github.com/LightDXY/BootMAE)
* [Lipschitz Continuity Retained Binary Neural Network](https://arxiv.org/abs/2207.06540)<br>:star:[code](https://github.com/42Shawn/LCR_BNN)
* [NeFSAC: Neurally Filtered Minimal Samples](https://arxiv.org/abs/2207.07872)<br>:star:[code](https://github.com/cavalli1234/NeFSAC)
* [Towards Understanding The Semidefinite Relaxations of Truncated Least-Squares in Robust Rotation Search](https://arxiv.org/abs/2207.08350)
* [Latency-Aware Collaborative Perception](https://arxiv.org/abs/2207.08560)<br>:star:[code](https://github.com/MediaBrain-SJTU/SyncNet)
* [MHR-Net: Multiple-Hypothesis Reconstruction of Non-Rigid Shapes from 2D Views](https://arxiv.org/abs/2207.09086)
* [SelectionConv: Convolutional Neural Networks for Non-rectilinear Image Data](https://arxiv.org/abs/2207.08979)
* [Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain](https://arxiv.org/abs/2207.10002)<br>:star:[code](https://github.com/boschresearch/sourcegen)
* [Discrete-Constrained Regression for Local Counting Models](https://arxiv.org/abs/2207.09865)
* [On the Versatile Uses of Partial Distance Correlation in Deep Learning](https://arxiv.org/abs/2207.09684)<br>:star:[code](https://github.com/zhenxingjian/Partial_Distance_Correlation)
* [Streamable Neural Fields](https://arxiv.org/abs/2207.09663)<br>:star:[code](https://github.com/jwcho5576/streamable_nf)
* [Contributions of Shape, Texture, and Color in Visual Recognition](https://arxiv.org/abs/2207.09510)<br>:star:[code](https://github.com/gyhandy/Humanoid-Vision-Engine)
* [Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model](https://arxiv.org/abs/2207.10040)
* [Latent Discriminant deterministic Uncertainty](https://arxiv.org/abs/2207.10130)<br>:star:[code](https://github.com/ENSTA-U2IS/LDU)
* [SPIN: An Empirical Evaluation on Sharing Parameters of Isotropic Networks](https://arxiv.org/abs/2207.10237)<br>:star:[code](https://github.com/apple/ml-spin)
* [UFO: Unified Feature Optimization](https://arxiv.org/abs/2207.10341)<br>:star:[code](https://github.com/PaddlePaddle/VIMER/tree/main/UFO)
* [POP: Mining POtential Performance of new fashion products via webly cross-modal query expansion](https://arxiv.org/abs/2207.11001)<br>:star:[code](https://github.com/HumaticsLAB/POP-Mining-POtential-Performance)
* [My View is the Best View: Procedure Learning from Egocentric Videos](https://arxiv.org/abs/2207.10883)<br>:star:[code](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning):house:[project](https://sid2697.github.io/egoprocel/)
* [Equivariance and Invariance Inductive Bias for Learning from Insufficient Data](https://arxiv.org/abs/2207.12258)<br>:star:[code](https://github.com/Wangt-CN/EqInv)
* [Contrastive Monotonic Pixel-Level Modulation](https://arxiv.org/abs/2207.11517)<br>:open_mouth:oral:star:[code](https://github.com/lukun199/MonoPix)
* [Neural-Sim: Learning to Generate Training Data with NeRF](https://arxiv.org/abs/2207.11368)<br>:star:[code](https://github.com/gyhandy/Neural-Sim-NeRF)
* [Learning Hierarchy Aware Features for Reducing Mistake Severity](https://arxiv.org/abs/2207.12646)<br>:star:[code](https://github.com/07Agarg/HAF)
* [Translating a Visual LEGO Manual to a Machine-Executable Plan](https://arxiv.org/abs/2207.12572)<br>:star:[code](https://github.com/Relento/lego_release):house:[project](https://cs.stanford.edu/~rcwang/projects/lego_manual/)
* [Hardly Perceptible Trojan Attack against Neural Networks with Bit Flips](https://arxiv.org/abs/2207.13417)<br>:star:[code](https://github.com/jiawangbai/HPT)
* [LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity](https://arxiv.org/abs/2207.13129)
* [MonteBoxFinder: Detecting and Filtering Primitives to Fit a Noisy Point Cloud](https://arxiv.org/abs/2207.14268)<br>:star:[code](https://github.com/MichaelRamamonjisoa/MonteBoxFinder):house:[project](https://michaelramamonjisoa.github.io/projects/MonteBoxFinder)
* [Neural Strands: Learning Hair Geometry and Appearance from Multi-View Images](https://arxiv.org/abs/2207.14067)<br>:house:[project](https://radualexandru.github.io/neural_strands/)
* [A Repulsive Force Unit for Garment Collision Handling in Neural Networks](https://arxiv.org/abs/2207.13871)<br>:house:[project](https://gamma.umd.edu/researchdirections/mlphysics/refu/)
* [Minimal Neural Atlas: Parameterizing Complex Surfaces with Minimal Charts and Distortion](https://arxiv.org/abs/2207.14782)<br>:star:[code](https://github.com/low5545/minimal-neural-atlas)
* [Revisiting the Critical Factors of Augmentation-Invariant Representation Learning](https://arxiv.org/abs/2208.00275)<br>:star:[code](https://github.com/megvii-research/revisitAIRL)
* [Fast Two-step Blind Optical Aberration Correction](https://arxiv.org/abs/2208.00950)<br>:star:[code](https://github.com/teboli/fast_two_stage_psf_correction)
* [Transformers as Meta-Learners for Implicit Neural Representations](https://arxiv.org/abs/2208.02801)<br>:star:[code](https://github.com/yinboc/trans-inr):house:[project](https://yinboc.github.io/trans-inr/)
* [Neighborhood Collective Estimation for Noisy Label Identification and Correction](https://arxiv.org/abs/2208.03207)<br>:star:[code](https://github.com/lijichang/LNL-NCE)
* [Rethinking Robust Representation Learning Under Fine-grained Noisy Faces](https://arxiv.org/abs/2208.04352)
* [Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast](https://arxiv.org/abs/2208.04378)<br>:star:[code](https://github.com/zhaodongsun/contrast-phys)
* [RelPose: Predicting Probabilistic Relative Rotation for Single Objects in the Wild](https://arxiv.org/abs/2208.05963)<br>:house:[project](https://jasonyzhang.com/relpose/)  
* [PRIF: Primary Ray-based Implicit Function](https://arxiv.org/abs/2208.06143)<br>:house:[project](https://augmentariumlab.github.io/PRIF/)
* [Context-Aware Streaming Perception in Dynamic Environments](https://arxiv.org/abs/2208.07479)<br>:star:[code](https://github.com/EyalSel/Contextual-Streaming-Perception)  
* [AdaBin: Improving Binary Neural Networks with Adaptive Binary Sets](https://arxiv.org/abs/2208.08084)
* [TRoVE: Transforming Road Scene Datasets into Photorealistic Virtual Environments](https://arxiv.org/abs/2208.07943)<br>:star:[code](https://github.com/shubham1810/trove)
* [L3: Accelerator-Friendly Lossless Image Format for High-Resolution, High-Throughput DNN Training](https://arxiv.org/abs/2208.08711)<br>:star:[code](https://github.com/SNU-ARC/L3)
* [GCISG: Guided Causal Invariant Learning for Improved Syn-to-real Generalization](https://arxiv.org/abs/2208.10024)
* [Learning Continuous Implicit Representation for Near-Periodic Patterns](https://arxiv.org/abs/2208.12278)<br>:star:[code](https://github.com/ArmastusChen/Learning-Continuous-Implicit-Representation-for-Near-Periodic-Patterns):house:[project](https://armastuschen.github.io/projects/NPP_Net/)
* [A Deep Moving-camera Background Model](https://arxiv.org/abs/2209.07923)<br>:star:[code](https://github.com/BGU-CS-VIL/DeepMCBM)
* [NashAE: Disentangling Representations through Adversarial Covariance Minimization](https://arxiv.org/abs/2209.10677)<br>:star:[code](https://github.com/ericyeats/nashae-beamsynthesis)
* [FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion](https://arxiv.org/abs/2209.11277)
* [Diversified Dynamic Routing for Vision Tasks](https://arxiv.org/abs/2209.13071)
* [Fast-ParC: Position Aware Global Kernel for ConvNets and ViTs](https://arxiv.org/abs/2210.04020)
* [Improving the Reliability for Confidence Estimation](https://arxiv.org/abs/2210.06776)
* [Attaining Class-level Forgetting in Pretrained Model using Few Samples](https://arxiv.org/abs/2210.10670)
* [Overexposure Mask Fusion: Generalizable Reverse ISP Multi-Step Refinement](https://arxiv.org/abs/2210.11511)
* [Photo-realistic Neural Domain Randomization](https://arxiv.org/abs/2210.12682)
* [Editable indoor lighting estimation](https://arxiv.org/abs/2211.03928)<br>:house:[project](https://lvsn.github.io/EditableIndoorLight/)
* [A Kendall Shape Space Approach to 3D Shape Estimation from 2D Landmarks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620354.pdf)
* [DeepShadow: Neural Shape from Shadow](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620403.pdf)<br>:star:[code](https://github.com/asafkar/deep_shadow):house:[project](https://asafkar.github.io/deepshadow/)
* [Intrinsic Neural Fields: Learning Functions on Manifolds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620609.pdf)
* [A Level Set Theory for Neural Implicit Evolution under Explicit Flows](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620699.pdf)
* [Unsupervised Pose-Aware Part Decomposition for Man-Made Articulated Objects](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630549.pdf)
* [MeshUDF: Fast and Differentiable Meshing of Unsigned Distance Field Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630566.pdf)
* [S2N: Suppression-Strengthen Network for Event-Based Recognition under Variant Illuminations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630701.pdf)<br>:star:[code](https://github.com/wanzengy/S2N-Suppression-Strengthen-Network)
* [A Spectral View of Randomized Smoothing under Common Corruptions: Benchmarking and Improving Certified Robustness](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640645.pdf)
* [Transform Your Smartphone into a DSLR Camera: Learning the ISP in the Wild](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660614.pdf)<br>:star:[code](https://github.com/4rdhendu/TransformPhone2DSLR)  
* [Data Association between Event Streams and Intensity Frames under Diverse Baselines](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670071.pdf)
* [Instance Contour Adjustment via Structure-Driven CNN](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670142.pdf)
* [3D Scene Inference from Transient Histograms](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670394.pdf)
* [Neural Space-Filling Curves](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670412.pdf)<br>:house:[project](https://hywang66.github.io/publication/neuralsfc)
* [LWGNet – Learned Wirtinger Gradients for Fourier Ptychographic Phase Retrieval](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670515.pdf)<br>:star:[code](https://github.com/at3e/LWGNet.git)
* [PANDORA: Polarization-Aided Neural Decomposition of Radiance](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670531.pdf)
* [Benchmarking Omni-Vision Representation through the Lens of Visual Realms](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670587.pdf)<br>:house:[project](https://zhangyuanhan-ai.github.io/OmniBenchmark)
* [When Deep Classifiers Agree: Analyzing Correlations between Learning Order and Image Statistics](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680388.pdf)
* [MUGEN: A Playground for Video-Audio-Text Multimodal Understanding and GENeration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680421.pdf)<br>:house:[project](https://mugen-org.github.io/)
* [The Missing Link: Finding Label Relations across Datasets](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680530.pdf)
* [Domain Adaptive Hand Keypoint and Pixel Localization in the Wild](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690070.pdf)<br>:house:[project](https://tkhkaeio.github.io/projects/22-hand-ps-da/)
* [DFNet: Enhance Absolute Pose Regression with Direct Feature Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700001.pdf)<br>:house:[project](https://code.active.vision/)  
* [GTCaR: Graph Transformer for Camera Re-Localization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700227.pdf)
* [Is Geometry Enough for Matching in Visual Localization?](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700402.pdf)<br>:star:[code](https://github.com/dvl-tum/gomatch)
* [Reducing Information Loss for Spiking Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710036.pdf)
* [Deep Partial Updating: Towards Communication Efficient Updating for On-Device Inference](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710137.pdf)
* [SP-Net: Slowly Progressing Dynamic Inference Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710225.pdf)
* [Meta-GF: Training Dynamic-Depth Neural Networks Harmoniously](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710691.pdf)<br>:star:[code](https://github.com/SYVAE/MetaGF)
* [You Already Have It: A Generator-Free Low-Precision DNN Training Framework Using Stochastic Rounding](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720034.pdf)
* [Real Spike: Learning Real-Valued Spikes for Spiking Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720052.pdf)
* [Exploring Lottery Ticket Hypothesis in Spiking Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720101.pdf)<br>:star:[code](https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Lottery-Ticket-Hypothesis-in-SNNs)
* [On the Angular Update and Hyperparameter Tuning of a Scale-Invariant Network](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720120.pdf)
* [LANA: Latency Aware Network Acceleration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720136.pdf)
* [Understanding the Dynamics of DNNs Using Graph Modularity](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720224.pdf)<br>:star:[code](https://github.com/yaolu-zjut/Dynamic-Graphs-Construction)
* [MIME: Minority Inclusion for Majority Group Enhancement of AI Performance](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730327.pdf)<br>:house:[project](https://visual.ee.ucla.edu/mime.htm/)
* [Trust, but Verify: Using Self-Supervised Probing to Improve Trustworthiness](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730362.pdf)<br>:star:[code](https://github.com/d-ailin/SSProbing)
* [Learning to Censor by Noisy Sampling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730378.pdf)
* [Anti-Neuron Watermarking: Protecting Personal Data against Unauthorized Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730449.pdf)
* [Recover Fair Deep Classification Models via Altering Pre-trained Structure](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730482.pdf) 
* [Decouple-and-Sample: Protecting Sensitive Information in Task Agnostic Data Release](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730499.pdf)<br>:star:[code](https://github.com/splitlearning/sanitizer)
* [Latent Space Smoothing for Individually Fair Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730535.pdf)
* [Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730554.pdf)<br>:star:[code](https://github.com/tochris/pts-uncertainty)
* [Image-Based CLIP-Guided Essence Transfer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730693.pdf)<br>:star:[code](https://github.com/hila-chefer/TargetCLIP)
* [End-to-End Visual Editing with a Generatively Pre-trained Artist](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750018.pdf)<br>:house:[project](https://www.robots.ox.ac.uk/~abrown/E2EVE/)
* [Sobolev Training for Implicit Neural Representations with Approximated Image Derivatives](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750070.pdf)<br>:star:[code](https://github.com/megvii-research/Sobolev_INRs)
* [L-Tracing: Fast Light Visibility Estimation on Neural Surfaces by Sphere Tracing](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750214.pdf)
* [Temporal-MPI: Enabling Multi-Plane Images for Dynamic Scene Modelling via Temporal Basis Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750321.pdf)
* [3D-Aware Semantic-Guided Generative Model for Human Synthesis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750337.pdf)<br>:house:[project](https://github.com/zhangqianhui/3DSGAN)
* [Unified Implicit Neural Stylization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750633.pdf)<br>:house:[project](https://zhiwenfan.github.io/INS)
* [Deep Portrait Delighting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760402.pdf)<br>:house:[project](https://www.wgtn.ac.nz/cmic)
* [Free-Viewpoint RGB-D Human Performance Capture and Rendering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760452.pdf)<br>:house:[project](https://www.phongnhhn.info/HVS_Net/)
* [Multiview Regenerative Morphing with Dual Flows](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760469.pdf)<br>:star:[code](https://github.com/jimtsai23/MorphFlow)
* [NeRF for Outdoor Scene Relighting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760593.pdf)<br>:house:[project](https://4dqv.mpi-inf.mpg.de/NeRF-OSR/)
* [Intelli-Paint: Towards Developing More Human-Intelligible Painting Agents](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760662.pdf)
* [Motion Transformer for Unsupervised Image Animation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760679.pdf)
* [NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760697.pdf)
* [Implicit Neural Representations for Variable Length Human Motion Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770359.pdf)<br>:star:[code](https://github.com/PACerv/ImplicitMotion)
* [Learning Object Placement via Dual-Path Graph Completion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770376.pdf)
* [Compositional Visual Generation with Composable Diffusion Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770426.pdf)<br>:house:[project](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/)
* [Spatial-Frequency Domain Information Integration for Pan-Sharpening](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780268.pdf)
* [ReCoNet: Recurrent Correction Network for Fast and Efficient Multi-Modality Image Fusion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780528.pdf)<br>:star:[code](https://github.com/dlut-dimt/reconet)
* [Rethinking Generic Camera Models for Deep Single Image Camera Calibration to Recover Rotation and Fisheye Distortion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780668.pdf)
* [Modeling Mask Uncertainty in Hyperspectral Image Reconstruction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790109.pdf)<br>:star:[code](https://github.com/Jiamian-Wang/mask_uncertainty_spectral_SCI)
* [Deep Fourier-Based Exposure Correction Network with Spatial-Frequency Interaction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790159.pdf)<br>:star:[code](https://github.com/KevinJ-Huang/FECNet)  
* [Towards Real-World HDRTV Reconstruction: A Data Synthesis-Based Approach](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790195.pdf)
* [Attention-Aware Learning for Hyperparameter Prediction in Image Processing Pipelines](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790265.pdf)
* [Memory-Augmented Model-Driven Network for Pansharpening](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790299.pdf)<br>:star:[code](https://github.com/Keyu-Yan/MMNet)
* [All You Need Is RAW: Defending against Adversarial Attacks with Camera Image Pipelines](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790316.pdf)
* [GRIT-VLP: Grouped Mini-Batch Sampling for Efficient Vision and Language Pre-training](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790386.pdf)<br>:star:[code](https://github.com/jaeseokbyun/GRIT-VLP)
* [Transformer with Implicit Edges for Particle-Based Physics Simulation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790539.pdf)<br>:star:[code](https://github.com/ftbabi/TIE_ECCV2022)
* [LA3: Efficient Label-Aware AutoAugment](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810258.pdf)
* [BA-Net: Bridge Attention for Deep Convolutional Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810293.pdf)<br>:star:[code](https://github.com/zhaoy376/Bridge-Attention)
* [SAU: Smooth Activation Function Using Convolution with Approximate Identities](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810309.pdf)
* [Almost-Orthogonal Layers for Efficient General-Purpose Lipschitz Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810345.pdf)<br>:star:[code](https://github.com/berndprach/AOL)
* [DLME: Deep Local-Flatness Manifold Embedding](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810569.pdf)
* [Accurate Detection of Proteins in Cryo-Electron Tomograms from Sparse Labels](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810636.pdf)
* [Social ODE: Multi-agent Trajectory Forecasting with Neural Ordinary Differential Equations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820211.pdf)
* [Entropy-Driven Sampling and Training Scheme for Conditional Diffusion Generation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820730.pdf)<br>:star:[code](https://github.com/ZGCTroy/ED-DPM)
* [Geometry-Guided Progressive NeRF for Generalizable and Efficient Neural Human Rendering](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830224.pdf)
* [Controllable Shadow Generation Using Pixel Height Maps](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830240.pdf)
* [Subspace Diffusion Generative Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830274.pdf)<br>:star:[code](https://github.com/bjing2016/subspace-diffusion)
* [MINER: Multiscale Implicit Neural Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830308.pdf)<br>:house:[project](https://vishwa91.github.io/miner)
* [An Embedded Feature Whitening Approach to Deep Neural Network Optimization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830324.pdf)<br>:star:[code](https://github.com/Yonghongwei/W-SGDM-and-W-Adam)
* [Q-FW: A Hybrid Classical-Quantum Frank-Wolfe for Quadratic Binary Optimization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830341.pdf)
* [Scalable Learning to Optimize: A Learned Optimizer Can Train Big Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830376.pdf)<br>:star:[code](https://github.com/VITA-Group/Scalable-L2O)
* [QISTA-ImageNet: A Deep Compressive Image Sensing Framework Solving ℓq-Norm Optimization Problem](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830394.pdf)
* [Rethinking Confidence Calibration for Failure Prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850512.pdf)<br>:star:[code](https://github.com/Impression2805/FMFP)
* [PRIME: A Few Primitives Can Boost Robustness to Common Corruptions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850615.pdf)<br>:star:[code](https://github.com/amodas/PRIME-augmentations)
* [Learning with Noisy Labels by Efficient Transition Matrix Estimation to Combat Label Miscorrection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850700.pdf)
* [Learning to Drive by Watching YouTube Videos: Action-Conditioned Contrastive Policy Pretraining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860109.pdf)<br>:house:[project](https://metadriverse.github.io/ACO/)
* [Balancing between Forgetting and Acquisition in Incremental Subpopulation Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860354.pdf)<br>:star:[code](https://github.com/wuyujack/ISL)
* [Sound Localization by Self-Supervised Time Delay Estimation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860476.pdf)<br>:house:[project](https://ificl.github.io/stereocrw/)
* [X-Learner: Learning Cross Sources and Tasks for Universal Visual Representation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860495.pdf)
* [A Contrastive Objective for Learning Disentangled Representations](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860566.pdf)<br>:star:[code](https://github.com/jonkahana/DCoDR)
* [A Gyrovector Space Approach for Symmetric Positive Semi-Definite Matrix Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870052.pdf)
* [Trading Positional Complexity vs Deepness in Coordinate Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870142.pdf)<br>:house:[project](https://osiriszjq.github.io/complex_encoding)
* [TO-Scene: A Large-Scale Dataset for Understanding 3D Tabletop Scenes](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870334.pdf)<br>:star:[code](https://github.com/GAP-LAB-CUHK-SZ/TO-Scene)
* [Primitive-Based Shape Abstraction via Nonparametric Bayesian Inference](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870472.pdf)
* [S2Net: Stochastic Sequential Pointcloud Forecasting](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870541.pdf)
* [LaLaLoc++: Global Floor Plan Comprehension for Layout Localisation in Unvisited Environments](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870681.pdf)
* [Variance-Aware Weight Initialization for Point Convolutional Neural Networks](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880073.pdf)
* [AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-Shot Interactions](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890089.pdf)<br>:house:[project](https://hyperplane-lab.github.io/AdaAfford)

  
扫码CV君微信（注明：CVPR）入微信交流群：
![image](https://user-images.githubusercontent.com/62801906/178399331-6a7c8997-b0d0-49a1-8fd7-4f1202d46382.png)
