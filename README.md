# [CEMTNet: A Cognitive-Emotion-Modulated Trend Network for Multimodal Depression Recognition](http://)

[**Yujie Huo, Weng Howe Chan, Ahmad Najmi Bin Amerhaider Nuar and Hongyu Gao**](http://)


This project implements a multimodal depression prediction model using:
- Text (ALBERT)
- Audio (Wav2Vec2)

## Clone demo code
```text
cd /workspace
git clone https://github.com/ghy-yhg/CEMTNet
```
## Dataset Setup

The DAIC-WOZ dataset needs to be used by logging into the official  [website](https://dcapswoz.ict.usc.edu/) and filling out an application form.
Unzip the files and place them as follows:


## 📦 Dataset Format

Expected structure (for both DAIC-WOZ and EATD):

```
data/
├── daic_woz/
│   ├── audio/            
│   ├── transcripts/        
│   └── labels.csv
├── eatd/
│   ├── audio/
│   ├── transcripts/
│   └── labels.csv
```
## Config Introduction
```text
pip install -r requirements.txt
```
### labels.csv format:

```
session_id,text_path,audio_path,label
P3001,transcripts/P3001.txt,audio/P3001.wav,1
...
```

## 🚀 Train

```bash
python train.py
```

## 🧪 Test

```bash
python test.py


```
## Issue
If there is an issue, please send an email to this address，yjh.academic@gmail.com

##  Citation
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:

```

@article{journal,
  author    = {Yujie Huo and Weng Howe Chan and Ahmad Najmi Bin Amerhaider Nuar and Hongyu Gao},
  title     = {CEMTNet: A Cognitive-Emotion-Modulated Trend Network for Multimodal Depression Recognition},
  journal   = {},
  year      = {2025},
  doi       = {We will update the above DOI after the paper is officially published}
}
```
