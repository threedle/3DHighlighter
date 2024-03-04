# 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions [CVPR 2023]

*[Dale Decatur](https://ddecatur.github.io/), [Itai Lang](https://itailang.github.io/), [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)*

University of Chicago

Abstract: *We present 3D Highlighter, a technique for localizing semantic regions on a mesh using text as input. A key feature of our system is the ability to interpret “out-of-domain” localizations. Our system demonstrates the ability to reason about where to place non-obviously related concepts on an input 3D shape, such as adding clothing to a bare 3D animal model. Our method contextualizes the text description using a neural field and colors the corresponding region of the shape using a probability-weighted blend. Our neural optimization is guided by a pre-trained CLIP encoder, which bypasses the need for any 3D datasets or 3D annotations. Thus, 3D Highlighter is highly flexible, general, and capable of producing localizations on a myriad of input shapes.*


<!-- ### [[Project Page](https://threedle.github.io/3DHighlighter/)] [[ArXiv](https://arxiv.org/abs/2212.11263)] -->
<a href="https://arxiv.org/abs/2212.11263"><img src="https://img.shields.io/badge/arXiv-3DHighlighter-b31b1b.svg" height=22.5></a>
<a href="https://threedle.github.io/3DHighlighter"><img src="https://img.shields.io/website?down_color=lightgrey&down_message=offline&label=Project%20Page&up_color=lightgreen&up_message=online&url=https%3A%2F%2Fpals.ttic.edu%2Fp%2Fscore-jacobian-chaining" height=22.5></a>

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
# hat on a candle
./demo/run_candle_hat.sh
# hat on a dog
./demo/run_dog_hat.sh
# shoes on a horse
./demo/run_horse_shoes.sh
```

### Note on Reproducibility
Due to small non-determinisms in CLIP's backward process and the sensitivity of our optimization, results can vary across different runs even when fully seeded. If the result of the optimization does not match the expected result, try re-running the optimization.

## Tips for Troubleshooting New Mesh+Region Combinations:
- Due to the sensitivity of the optimization process, if a mesh+prompt combination does not work on the first try, rerun the optimization with a new seed as it might just have found a bad local minimum.
- If an initial specification of a region does not work well, try describing that region with more specific language (i.e. 'eyeglasses' instead of 'glasses'). Also, try using a different target localization text that might correspond to a similar region (i.e. using 'headphones' or 'earmuffs' instead of 'ears').
- In our experiments, we found that using gray and highlighter colors and the prompt format of `"A 3D render of a gray [object] with highlighted [region]"` works best for most mesh+region combinations. However, we encourage users to edit the code to try different prompt specifications since different wordings might work better with new and different mesh+region combinations.
- The traingulation of the mesh is important. Meshes containing long, skinny triangles and/or small numbers of vertices can lead to bad optimizations.

## Citation
```
@inproceedings{decatur2023highlighter,
  title={3d highlighter: Localizing regions on 3d shapes via text descriptions},
  author={Decatur, Dale and Lang, Itai and Hanocka, Rana},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20930--20939},
  year={2023}
}
```
