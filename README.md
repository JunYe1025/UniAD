<div align="center">   
  
# (WIP) Planning-oriented Autonomous Driving
</div>

<!-- <p align="center">
 <a href="https://opendrivelab.github.io/UniAD/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Open-yellowgreen.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenDriveLab/UniAD/blob/master/LICENSE">
    <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" target="_blank" />
  </a>
  <a href="https://github.com/OpenDriveLab/UniAD/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
    <img alt="Good first issue" src="https://img.shields.io/github/issues/OpenDriveLab/UniAD/good%20first%20issue" target="_blank" />
  </a>
</p> -->

<h3 align="center">
  <a href="https://opendrivelab.github.io/UniAD/">project page</a> |
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="">video</a> |
  <a href="https://opendrivelab.com/">OpenDriveLab</a>
  
</h3>

https://user-images.githubusercontent.com/48089846/202974395-15fe83ac-eebb-4f38-8172-b8ca8c65127e.mp4

This repository will host the code of UniAD.

> [Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)
>
> Yihan Hu*, Jiazhi Yang*, [Li Chen*](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en&authuser=1), Keyu Li*, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, [Hongyang Li](https://lihongyang.info/)
> - CVPR 2023, award candidate
> - Primary contact: Li Chen ( lichen@pjlab.org.cn )

![teaser](sources/pipeline.png)

## Highlights

- :oncoming_automobile: **Planning-oriented philosophy**: UniAD is a Unified Autonomous Driving algorithm framework devised following a planning-oriented philosophy. Instead of standalone modular design and multi-task learning, perception, prediciton and planning tasks/components should opt in and be prioritized hierarchically, and we demonstrate the performance can be enhanced to a new level.
- :trophy: **SOTA performance**: All tasks among UniAD achieve SOTA performance, especially prediction and planning (motion: 0.71m minADE, occ: 63.4% IoU-n., plan: 0.31% avg.Col)

## News

- Code & model release: We are actively re-organizing the codebase for better readability. The estimated time is late March. Please stay tuned!
- About the title: To avoid misunderstanding about the "goal", we change the title from "Goal-oriented" to "Planning-oriented" as is suggested by the reviewers. We originally use  "goal" to indicate the  final safe planning in an AD pipeline, rather than "goal-point" -- the destination of a sequence of actions.
- [2023/03/21] :rocket::rocket: UniAD paper is accepted by CVPR 2023, as an **award candidate** (12 out of 9155 submissions and 2360 accepted papers)!
- [2022/12/21] UniAD [paper](https://arxiv.org/abs/2212.10156) is available on arXiv!

<!-- 
## Getting started

- [Installation]()
- [Dataset preparation]()
- [Train and eval]()
-->

## Getting Started
* Step 1. [Installation](docs/INSTALL.md)
* Step 2. [Prepare Dataset](docs/DATA_PREP.md)
* Step 3. [Train and Eval](docs/TRAIN_EVAL.md)


## Models
### Stage-one: Perception training

| Method | Encoder | Tracking<br>AMOTA | Mapping<br>IoU-lane | config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| 
| UniAD-S | R50 | -  | - | TBA | TBA |
| UniAD-B | R101 | 0.390 | 0.297 |  [base-stage1](projects/configs/track_map/base_stage1.py) | [base-stage1](https://github.com/OpenDriveLab/UniAD/releases/download/untagged-d7e1d5e20eded789eee9/uniad_base_track_map.pth) |
| UniAD-L | V2-99 | - | - | TBA | TBA |



### Stage-two: End-to-end training

Pre-trained models and results under main metrics are provided below. We refer you to the [paper](https://arxiv.org/abs/2212.10156) for more details.

| Method | Encoder | Tracking<br>AMOTA | Mapping<br>IoU-lane | Motion<br>minADE |Occupancy<br>IoU-n. | Planning<br>avg.Col. | config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: | :---: |
| UniAD-S | R50 | 0.241  | 0.315 | 0.788 | 59.4  | 0.32 | TBA | TBA |
| UniAD-B | R101 | 0.359 | 0.313 | 0.708 | 63.4 | 0.31 |  TBA | TBA |
| UniAD-L | V2-99 | 0.409 | 0.323 | 0.723 | 64.1 | 0.29 | TBA | TBA |



## Todo list
- [ ] (Long-term) Improve flexibility for future extensions
- [ ] Support bs>1
- [ ] All checkpoints
- [ ] Base-model checkpoints
- [x] Code initialization


## License

All assets and code are under the [Apache 2.0 license](https://github.com/OpenDriveLab/UniAD/blob/master/LICENSE) unless specified otherwise.

## Citation

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@inproceedings{uniad,
 title={Planning-oriented Autonomous Driving}, 
 author={Yihan Hu and Jiazhi Yang and Li Chen and Keyu Li and Chonghao Sima and Xizhou Zhu and Siqi Chai and Senyao Du and Tianwei Lin and Wenhai Wang and Lewei Lu and Xiaosong Jia and Qiang Liu and Jifeng Dai and Yu Qiao and Hongyang Li},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 year={2023},
}
```
## Related resources

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (:rocket:Ours!)
- [ST-P3](https://github.com/OpenPerceptionX/ST-P3) (:rocket:Ours!)
- [FIERY](https://github.com/wayveai/fiery)
- [MOTR](https://github.com/megvii-research/MOTR)
- [BEVerse](https://github.com/zhangyp15/BEVerse)