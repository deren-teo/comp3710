# comp3710

Repository tracking source code for [COMP3710: Pattern Recognition and Analysis](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) coursework.

## Demo 1: Fractals

This set of Python notebooks explores fractals and symmetry using PyTorch over three parts:
- [Part 1](./demo1/part1.ipynb) -- serves as an introduction to PyTorch and implements a Gabor filter
- [Part 2](./demo1/part2.ipynb) -- explores the Mandelbrot set with the aid of PyTorch parallelisation
- [Part 3](./demo1/part3.ipynb) -- demonstrates a different fractal creation technique to visualise the [Heighway Dragon](https://en.wikipedia.org/wiki/Dragon_curve)

## Demo 2: Pattern Recognition

This set of Python notebooks moves on to exploring more recent pattern recognition techniques:

- [Part 1](./demo2/1_eigenfaces.ipynb) -- uses the eigenface method and a random forest model to classify faces from the [LFW dataset](http://vis-www.cs.umass.edu/lfw/)
- [Part 2a](./demo2/2a_lfw_cnn.ipynb) -- improves on the above classification result using a ResNet model
- [Part 2b](./demo2/2b_cifar_resnet.py) -- applies another ResNet for the CIFAR10 dataset to meet the [DAWNBench challenge](https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time)
- [Part 3](./demo2/3_celeba_gan.py) -- trains a GAN to generate images based on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## See Also

This course had three demonstrations, but the third involved forking and contributing to a separate public GitHub repository.

See the [`topic-recognition`](https://github.com/deren-teo/PatternAnalysis-2023/tree/topic-recognition/recognition/adni_vit_45285545) branch of the PatternAnalysis-2023 repository for an implementation of a Vision Transformer (ViT) trained on the dataset from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/). The ViT is intended to classify MRI brain images as either Cognitive Normal (CN) or representative of Alzheimer's disease (AD), and achieves 75.11% accuracy using transfer learning.
