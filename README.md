# 3D Highlighter [[Project Page](https://threedle.github.io/3DHighlighter/)]
<!-- [![arXiv](https://img.shields.io/badge/arXiv-3DHighlighter-b31b1b.svg)](https://arxiv.org/abs/2112.03221) -->
![Pytorch](https://img.shields.io/badge/PyTorch->=1.12.1-Red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA->=11.3.1-Red?logo=CUDA)


Public code release for "3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions".

## Getting Started
### Installation

```
conda env create --file 3DHighlighter.yml
conda activate 3DHighlighter
```

### System Requirements
- Python 3.7
- CUDA 11
- 16 GB GPU

### Run Examples
Run the scripts below to get example localizations.
```
# hat on a dog
./demo/run_dog_hat.sh
# shoes on a horse
./demo/run_horse_shoes.sh
# ...
```

### Note on Reproducibility
Due to the non-determinism of CLIP's backwards process, our results can vary even when fully seeded.
If the result of the optimizaiton does not match the expected result, try re-running the optimization.

## Citation
```
@article{decatur2022highlighter,
         author = {Decatur, Dale and Lang, Itai and Hanocka, Rana},
         title  = {3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions},
         journal = {arXiv},
         year = {2022}
}
```