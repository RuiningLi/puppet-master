# Puppet-Master
Official implementation of 'Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics'

<p align="center">
  [<a href="https://arxiv.org/pdf/2408.04631"><strong>arXiv</strong></a>]
  [<a href="https://huggingface.co/spaces/rayli/Puppet-Master"><strong>Demo</strong></a>]
  [<a href="https://vgg-puppetmaster.github.io/"><strong>Project</strong></a>]
  [<a href="#citation"><strong>BibTeX</strong></a>]
</p>

### News
- **2024-Sept-25** Training script released.
- **2024-Aug-17** Pre-trained checkpoints and demo released on Hugging Face. Check [here](https://huggingface.co/spaces/rayli/Puppet-Master) for demo and [here](https://huggingface.co/spaces/rayli/Puppet-Master/tree/main) for code.

### Examples

##### Man-Made Objects
![Man-Made Objects](https://vgg-puppetmaster.github.io/resources/manmade.gif)

##### Animals
![Animals](https://vgg-puppetmaster.github.io/resources/animal.gif)

##### Humans
![Humans](https://vgg-puppetmaster.github.io/resources/human.gif)

### Objaverse-Animation & Objaverse-Animation-HQ
See the `data` folder.

### Training
We provide a minimum viable training script to demonstrate how to use our dataset to fine-tune Stable Video Diffusion.

You can use the following command:
```
accelerate launch --num_processes 1 --mixed_precision fp16 train.py --config configs/train-puppet-master.yaml
```

To reduce the memory overhead, we cache all the latents and CLIP embeddings of the rendered frames.

Note this is only a working example. Our final model is trained using a combined dataset of Objaverse-Animation-HQ and [Drag-a-Move](https://github.com/RuiningLi/DragAPart/tree/main/Drag-a-Move).

### Inference
We provide an interactive demo [here](https://huggingface.co/spaces/rayli/Puppet-Master). Check it out!

### Evaluation
Our evaluation utilizes an unseen test set of [Drag-a-Move](https://github.com/RuiningLi/DragAPart/tree/main/Drag-a-Move), consisting of 100 examples.
The whole test set is provided in `DragAMove-test-batches` folder.
The test examples can be read directly from the `xxxxx.pkl` files and are in the same format as those loaded from `DragVideoDataset` implemented in `dataset.py`.

### TODO
- [x] Release pre-trained checkpoint & inference code.
- [x] Release training code.
- [x] Release Objaverse-Animation & Objaverse-Animation rendering script.

### Citation

```
@article{li2024puppetmaster,
  title     = {Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics},
  author    = {Li, Ruining and Zheng, Chuanxia and Rupprecht, Christian and Vedaldi, Andrea},
  journal   = {arXiv preprint arXiv:2408.04631},
  year      = {2024}
}
```
