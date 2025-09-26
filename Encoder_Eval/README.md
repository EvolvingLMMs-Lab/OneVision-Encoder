# Encoder_Eval: A Unified Evaluation Suite for Video and Image Encoders

This repository provides a unified evaluation framework for benchmarking **video** and **image** encoders across diverse tasks, including **linear probing**, **attentive probing**, **dense segmentation**, and **object detection**.

---

## 📅 Project Progress

### ✅ Completed
- [x] `video_attentive_probe`: Attention-based probing for video encoders
- [x] `video_linear_probe`: Linear probing for video encoders

### ⬜ Upcoming / In Development
- [ ] `image_attentive_probe`: Attention-based probing for image encoders
- [ ] `image_linear_probe`: Linear probing for image encoders
- [ ] `dense_segmentation`: Dense prediction benchmarking (image/video)
- [ ] `object_detection`: Detection task evaluation (image/video)

---

## 💡 Key Features
- Support for both **video** and **image** modalities.
- Modular design for easy integration of new probing techniques.
- Standardized evaluation pipelines for encoder representations.
- Designed to benchmark both **frozen** and **fine-tuned** encoders.

## 🔧 Setup

```bash
# Clone the repo
git clone git@github.com:FeilongTangmonash/Encoder_Eval.git
cd Encoder_Eval

# Install dependencies
pip install -r requirements.txt
```
## 🧱 code structure

<pre>
video_vit/
└── video_encoder_eval/
    └── video_linear_probe/
        └── checkpoint/
            └── mlcd_base/
                └── backbone_base224.pt
</pre>


## 🚀 Usage
We provide example scripts to perform a full evaluation of the UMT model using both the attentive probe and the linear probe methods. Simply run the commands below:
```
bash src/video_attentive_probe.sh
bash src/video_linear_probe.sh
```
