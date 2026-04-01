# EchoTraffic: Enhancing Traffic Anomaly Understanding with Audio-Visual Insights

[[📄 Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_EchoTraffic_Enhancing_Traffic_Anomaly_Understanding_with_Audio-Visual_Insights_CVPR_2025_paper.pdf) ｜ [[🤗 Model (EchoTraffic)]](https://huggingface.co/harryhsing/EchoTraffic-7B) ｜ [[🤗 Dataset (AV-TAU)]](https://huggingface.co/datasets/harryhsing/AV-TAU)

---

## 📦 Datase: AV-TAU
We release the **AV-TAU dataset** to support audio-visual traffic anomaly understanding.  

👉 Available on [AV-TAU](https://huggingface.co/datasets/harryhsing/AV-TAU)

---

## 🤖 Model: EchoTraffic
Pre-trained and Supervised Fine‑tuning (SFT) checkpoints for **EchoTraffic** are publicly available.

👉 Available on [EchoTraffic](https://huggingface.co/harryhsing/EchoTraffic-7B)

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/HarryHsing/EchoTraffic
cd EchoTraffic

# Create conda environment
conda env create -f environment.yml
conda activate echotraffic
```

---

## Inference

```bash
python inference.py \
    --video-path ./test_video/036680.mp4 \
    --prompt "What unusual event takes place in the video?" \
    --model-path ./ckpt/videollama_video_audio_sft/checkpoint_0.pth \
    --cfg-path ./eval_configs/finetune_eval.yaml \
    --gpu-id 0 \
    --num-frames 8
```

> Why did `num_frames=32` fail but `num_frames=8` work?
>  
> In this checkpoint/config, the audio-video fusion path is aligned for up to 8 temporal clips.  
> Setting `num_frames=32` can produce temporal shape mismatch errors (for example: `tensor a (8) vs tensor b (32)`), even without OOM.  
> Use `--num-frames 8` unless you also retrain/modify temporal embedding settings.

---

## Training
⚙️ Trained on A6000 (48G) GPUs.

### Pre‑training
```bash
torchrun --nproc_per_node=8 train.py --cfg-path  ./train_configs/video_audio_pretrain.yaml
```

### Supervised Fine‑tuning (SFT)
```bash
torchrun --nproc_per_node=4 train.py --cfg-path  ./train_configs/video_audio_finetune.yaml
```

---

## Preparation (Datasets and Weights)
You can download `EchoTraffic` from [Hugging Face](https://huggingface.co/harryhsing/EchoTraffic-7B).
### Model Weights
```bash
./ckpt/
├── llama-2-7b-chat-hf                        
├── videollama_video_audio_pretrain           # Pre‑training checkpoint 
├── videollama_video_audio_sft                # Supervised Fine‑tuning (SFT) checkpoint
└── imagebind_huge.pth

```

### Dataset: AV-TAU (for Supervised Fine‑Tuning / Inference)
You can download `AV-TAU` from [Hugging Face](https://huggingface.co/datasets/harryhsing/AV-TAU).
```bash
./datasets/AV-TAU/
├── videos/                                   # raw video files
├── annotations/
│   ├── sft_formatted_train.json              # annotation file
│   └── sft_formatted_test.json               # annotation file
```

### Dataset: InternVid-10M-FLT (for Pre‑training)
You can download `InternVid-10M-FLT` from [OpenDataLab](https://opendatalab.com/vd-foundation/InternVid-10M-FLT).
```bash
./datasets/vd-foundation___InternVid-10M-FLT/
├── InternVid-10M-FLT-INFO_with_audio.jsonl   # annotation file (exact name may vary)
└── raw/                                      # raw video files
```
---


## 📑 Citation
If you use this dataset or our paper, please cite:

```bibtex
@InProceedings{Xing_2025_CVPR,
    author    = {Xing, Zhenghao and Chen, Hao and Xie, Binzhu and Xu, Jiaqi and Guo, Ziyu and Xu, Xuemiao and Hao, Jianye and Fu, Chi-Wing and Hu, Xiaowei and Heng, Pheng-Ann},
    title     = {EchoTraffic: Enhancing Traffic Anomaly Understanding with Audio-Visual Insights},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {19098-19108}
}
