# 面向腔镜手术机器人的术前医学影像配准智能体

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.19-5D91D4.svg"></a>
<a href="https://github.com/modelscope/ms-swift"><img src="https://img.shields.io/badge/ms%20swift-3.6-green"></a>
</p>
<div align="center">
  <img src="https://github.com/hnuxyz/HNUMedAgent/blob/main/src/Fig/logo.png">
</div>

https://github.com/user-attachments/assets/dadee47b-294e-4421-a72e-0def372cd740

##  📖 目录

- [简介](#-简介)
- [环境配置](#-环境配置)
- [微调与部署](#-微调与部署)
- [License](#-license)

## 📝 简介
本智能体是笔者在完成硕士学位论文时所构建的第五章主体，本意是为了探索新一代智慧医疗技术与多模态视觉语言大模型。整个软件平台面向腹腔镜智能医疗手术机器人场景，平台功能包括一个基于Intern-S1-mini多模态视觉-语言大模型进行LoRA Plus微调的医疗场景交互大模型以及基于腹部CT配准的可视化应用。

🤟 感谢上海AI Lab提供的A100算力支持！

🧠 平台基于Intern-S1-mini多模态视觉-语言大模型、LMDeploy部署工具、ms-swift工具箱构建

<div align="center">
  <img src="https://github.com/hnuxyz/HNUMedAgent/blob/main/src/Fig/pipeline.png">
</div>

## 🛠️ 环境配置

- ### 大模型微调、部署环境

```bash
conda create -n ms-swift python=3.10
conda activate ms-swift
pip install ms-swift==3.6
pip install lmdeploy
pip install nibabel
```

- ### 配准环境

👋参考我们团队获得MICCAI Learn2Reg挑战赛国际亚军的解决方案[EOIR](https://github.com/XiangChen1994/EOIR)

## 🚀 微调与部署

- 准备[SLAKE](https://www.med-vqa.com/slake/)、[MediScope](https://huggingface.co/datasets/AQ-MedAI/PulseMind)数据集，并转化为ms-swift需求的格式；
- 下载EOIR腹部CT配准权重[![](https://img.shields.io/badge/google-link-red?logo=google)]()
- 启动微调：

### 第一阶段PEFT微调(使用SLAKE数据集)

- 将相关数据及基座模型配置好后，命令行进入SFT文件夹路径，即可在ms-swift中一键启动：

```bash
cd ./SFT
conda activate ms-swift
bash Slake_sft_InternS1.sh
```

- 微调完成后，选择想要合并权重的checkpoint，完成合并：

```bash
bash merge.sh
```

### 第二阶段PEFT微调(使用MediScope数据集)

- 将相关数据及基座模型配置好后，命令行进入SFT文件夹路径，即可在ms-swift中一键启动：

```
cd ./SFT
conda activate ms-swift
bash MediScope_sft_InternS1.sh
```

- 微调完成后，选择想要合并权重的checkpoint，完成合并：

```
bash merge.sh
```

### 启动API、Web UI服务

- 启动LMDeploy大模型API服务，实现KV Cache高效端侧部署：

```
python lmdeploy_server.py
```

- 本项目部署在上海AI Lab提供的A100工作站上，因此使用VS code或其他工具进行端口转发Gradio界面，实现快速前端交互：

```bash
VS code运行Agent_UI.py
```

- 使用VS code端口转发功能，默认是127.0.0.1:7860，端口号可自行在YueLu_Agent.py中修改合法值；
- 在本地浏览器中输入127.0.0.1:7860（根据自己部署的情况修改）
- 交互：

<div align="center">
  <img src="https://github.com/hnuxyz/HNUMedAgent/blob/main/src/Fig/UI.jpg">
</div>

## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License！
