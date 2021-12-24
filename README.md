![Logo](https://storage.googleapis.com/tf_model_garden/tf_model_garden_logo.png)

# Welcome to the Model Garden for TensorFlow

The TensorFlow Model Garden is a repository with a number of different
implementations of state-of-the-art (SOTA) models and modeling solutions for
TensorFlow users. We aim to demonstrate the best practices for modeling so that
TensorFlow users can take full advantage of TensorFlow for their research and
product development.

To improve the transparency and reproducibility of our models, training logs on
[TensorBoard.dev](https://tensorboard.dev) are also provided for models to the
extent possible though not all models are suitable.

| Directory | Description |
|-----------|-------------|
| [official](official) | • A collection of example implementations for SOTA models using the latest TensorFlow 2's high-level APIs<br />• Officially maintained, supported, and kept up to date with the latest TensorFlow 2 APIs by TensorFlow<br />• Reasonably optimized for fast performance while still being easy to read |
| [research](research) | • A collection of research model implementations in TensorFlow 1 or 2 by researchers<br />• Maintained and supported by researchers |
| [community](community) | • A curated list of the GitHub repositories with machine learning models and implementations powered by TensorFlow 2 |
| [orbit](orbit) | • A flexible and lightweight library that users can easily use or fork when writing customized training loop code in TensorFlow 2.x. It seamlessly integrates with `tf.distribute` and supports running on different device types (CPU, GPU, and TPU). |

## [Announcements](https://github.com/tensorflow/models/wiki/Announcements)

## Contributions

[![help wanted:paper implementation](https://img.shields.io/github/issues/tensorflow/models/help%20wanted%3Apaper%20implementation)](https://github.com/tensorflow/models/labels/help%20wanted%3Apaper%20implementation)

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).

## License

[Apache License 2.0](LICENSE)

## Citing TensorFlow Model Garden

If you use TensorFlow Model Garden in your research, please cite this repository.

```
@misc{tensorflowmodelgarden2020,
  author = {Hongkun Yu and Chen Chen and Xianzhi Du and Yeqing Li and
            Abdullah Rashwan and Le Hou and Pengchong Jin and Fan Yang and
            Frederick Liu and Jaeyoun Kim and Jing Li},
  title = {{TensorFlow Model Garden}},
  howpublished = {\url{https://github.com/tensorflow/models}},
  year = {2020}
}
```

#Getting started

## Drivers and packages
Make sure the correct nvidia gpu driver is installed as different come with different cuda versions
Install graphics 460 first for cuda 11.2

then install dev and runtime deb packages for cudnn 8.1

##Download model
Be sure to download the correct model from the Tensorflow model garden and point ssd_efficientdet_d0_512x512_coco17_tpu-8.config to the model.

## How to run tensorflow trained network
Make sure the training folder is empty before training
python3 model_main_tf2.py     --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config     --model_dir=training     --alsologtostderr

## Start website tracking the training process
tensorboard --logdir=training/train

## Start validation procces (During training, need more gpu memory to work)
python3 model_main_tf2.py \
  --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
  --model_dir=/home/jonfp/Downloads/centernet_hg104_512x512_coco17_tpu-8/saved_model \
  --checkpoint_dir=training \
  --num_workers=1 \
  --sample_1_of_n_eval_examples=1


## Export trained model:
python3 exporter_main_v2.py     --trained_checkpoint_dir=training     --pipeline_config_path=configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config     --output_directory inference_graph


## Use the model on webcam:
python3 stream.py -l /home/jonfp/racc/test/raccoons_label_map.pbtxt -m /home/jonfp/Downloads/models/research/object_detection/inference_graph/saved_model/

