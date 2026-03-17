# [NeurIPS 2025] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning


* **🧑‍💻 Authors**: [Xudong Yan](https://xud-yan.github.io/), [Songhe Feng](https://faculty.bjtu.edu.cn/8407/)
* **🏛️ Affiliations**: Beijing Jiaotong University
* **🔍 More details**: [[arXiv version]](https://arxiv.org/abs/2510.20162) | [[NeurIPS version]](https://neurips.cc/virtual/2025/loc/san-diego/poster/117606) | [[code]](https://github.com/xud-yan/TOMCAT)
* **🕰️ Latest Update Time**：2026.3.16 (Please use the newest yml file for C-GQA config. Sorry that the yml file for the C-GQA dataset was uploaded earlier contained suboptimal parameter settings, which has now been corrected (抱歉之前C-GQA数据集上传了并非最优的参数配置的yml，现已修正).)

## 📝 Overview

**TL;DR**: We introduce a new setting OTTA-CZSL, and propose to accumulate multimodal knowledge to overcome the label distribution shift caused by unseen compositions recombined from attributes and objects by leveraging unsupervised data at test time.

**We look forward to more work focusing on test-time strategies for addressing the CZSL task!**

<img src=".\images\overview.png" style="zoom: 50%;" />

## ⚙️ Setup

Our work is implemented in PyTorch framework. Create a conda environment `tomcat` using:

```
conda create --name tomcat python=3.8.0
conda activate tomcat
pip install -r requirements.txt
```



## ⬇️ Download

**Datasets**: In our work, we conduct experiments on four datasets: UT-Zappos, MIT-States, C-GQA, and Clothing16K. For Clothing16K, you can download this dataset from [this website](https://drive.google.com/drive/folders/1ZSw4uL8bjxKxBhrEFVeG3rgewDyDVIWj). For UT-Zappos, MIT-States and C-GQA, please using:

```
bash utils/download_data.sh
```



**Pre-trained models**: ViT-Large-Patch14 can be downloaded using the following command:

```
cd {CLIP_Model_Path}
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

```

## 🏋️ Training Phase

Train the Base Model of TOMCAT with a specified configure file using:

   ```
   python train.py --cfg config/{DATASET_NAME}.yml
   ```

   

## 📊 Test Time

Implement TOMCAT model using:

```
python test.py --cfg config/{DATASET_NAME}.yml
```

Note

## 📚 Citation

If you find our work helpful, please star the repository and cite our paper:

```
@article{Yan_2025_NeurIPS,
  title={TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning},
  author={Yan, Xudong and Feng, Songhe},
  journal={arXiv preprint arXiv:2510.20162},
  year={2025}
}
```
or
```
the NeurIPS version (will be released soon)
```



## 🙏 Acknowledgement

Thanks for the publicly available code of [AdapterFormer](https://github.com/ShoufaChen/AdaptFormer), [Troika](https://github.com/bighuang624/Troika), [TDA](https://github.com/kdiAAA/TDA), [TPS](https://github.com/elaine-sui/TPS),  and [DPE](https://github.com/zhangce01/DPE-CLIP).

## 📬 Contact

If you have any questions or are interested in collaboration, please feel free to contact me at xud_yan@bjtu.edu.cn .
