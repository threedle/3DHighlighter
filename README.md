# 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions

*[Dale Decatur](https://ddecatur.github.io/), [Itai Lang](https://scholar.google.com/citations?user=q0bBhtsAAAAJ&hl=en), [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)*

University of Chicago

Abstract: *We present 3D Highlighter, a technique for localizing semantic regions on a mesh using text as input. A key feature of our system is the ability to interpret “out-of-domain” localizations. Our system demonstrates the ability to reason about where to place non-obviously related concepts on an input 3D shape, such as adding clothing to a bare 3D animal model. Our method contextualizes the text description using a neural field and colors the corresponding region of the shape using a probability-weighted blend. Our neural optimization is guided by a pre-trained CLIP encoder, which bypasses the need for any 3D datasets or 3D annotations. Thus, 3D Highlighter is highly flexible, general, and capable of producing localizations on a myriad of input shapes.*


<!-- ### [[Project Page](https://threedle.github.io/3DHighlighter/)] [[ArXiv]()] -->
<a href=""><img src="https://img.shields.io/badge/arXiv-3DHighlighter-b31b1b.svg" height=22.5></a>
<a href="https://threedle.github.io/3DHighlighter"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A%2F%2Fpals.ttic.edu%2Fp%2Fscore-jacobian-chaining" height=22.5></a>
<!-- [![arXiv](https://img.shields.io/badge/arXiv-3DHighlighter-b31b1b.svg)]() -->
<!-- ![Pytorch](https://img.shields.io/badge/PyTorch->=1.12.1-Red?logo=pytorch) -->
<!-- ![CUDA](https://img.shields.io/badge/CUDA->=11.3.1-Red?logo=CUDA) -->

![teaser](./media/teaser.png)


## Installation

Install and activate the conda environment with the following commands. 
```
conda env create --file 3DHighlighter.yml
conda activate 3DHighlighter
```
Note: The installation will fail if run on something other than a CUDA GPU machine.

#### System Requirements
- Python 3.9
- CUDA 11
- 16 GB GPU

## Run Examples
Run the scripts below to get example localizations.
```
# hat on a dog
./demo/run_dog_hat.sh
# shoes on a horse
./demo/run_horse_shoes.sh
# hat on a candle
./demo/run_candle_hat.sh
```

### Note on Reproducibility
Due to small non-determinisms in CLIP's backward process and the sensitivity of our optimization, results can vary across different runs even when fully seeded. If the result of the optimization does not match the expected result, try re-running the optimization.


## Citation
```
@article{decatur2022highlighter,
    author = {Decatur, Dale and Lang, Itai and Hanocka, Rana},
    title  = {3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions},
    journal = {arXiv},
    year = {2022}
}
```
