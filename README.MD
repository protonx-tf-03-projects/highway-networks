# Highway Network

<p align="center">
    <img src='https://i.imgur.com/2cZqiA9.png' width=200 class="center" title="hover text">
</p>

Implementation for paper **[Highway Network (2015)](https://arxiv.org/abs/1505.00387)**. Give us a star if you like this repo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PRRdM24OTv24_h0jQXGbTJbiOVGqZe4V)

**Model architecture:**

<p align="center">
    <img src='https://i.imgur.com/54bFTRm.png' width=800 class="center">
</p>

_Additional references, experiments and analysis of the model is available in the **[full paper](http://arxiv.org/abs/1507.06228)** extending this study._
 
**Authors (Github & Email):**

- pnbl-123 (team leader) – [phungngbaolong@gmail.com](mailto:phungngbaolong@gmail.com)
- quan030994 – [tranquan030894@gmail.com](mailto:tranquan030894@gmail.com)
- hatruong29 – [maiha.th.92@gmail.com](mailto:maiha.th.92@gmail.com)

**Advisors:**

- bangoc123 – [protonxai@gmail.com](mailto:protonxai@gmail.com)

This library belongs to our project: Protonx-tf-03-projects where we implement AI papers and publish all source codes.

## I. Set up environment

1. Make sure you have installed Miniconda. If not yet, see the setup document [here](https://docs.conda.io/en/latest/miniconda.html).
2. `cd` into `highway-networks` and use command line `conda env create -f environment.yml` to setup the environment
3. Run conda environment using the command `conda activate highway-networks`

## II. Set up your dataset

The dataset used for Highway Networks is the MNIST database of handwritten digits, available from this page. It has a training set of `60000` examples, and a test set of `10000` examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

## III. Training Process

Training script:

```python
python train.py  --t-bias ${t_bias} --acti-h ${acti_h} --acti-t ${acti_t} --num-classes ${num_classes} --num-of-layers ${num_of_layers} --batch-size ${batch-size} --epochs ${epochs}

```

Example: You want to train a highway network for 10 classes with the bias of -2.0, 10 layers, and batch size of 128 in 10 epochs

```python
!python train.py --t-bias -2.0 --acti-h tf.nn.relu --acti-t tf.nn.sigmoid --num-of-layers 10 --batch-size 128 --num-classes 10 --epochs 10 

```

There are some important arguments for the script you should consider when running it:

- `num_classes`: the number of your problem classes / output categories
- `num_of_layers`: the number of layer in network
- `t_bias`: the bias in the transform gate which can be initialized with an negative value such that the network is initially biased towards carry behavior.
- `acti_h`: the activation function following the transform function h (ReLU or Tanh)
- `acti_t`: the activation function following the transform gate t (Sigmoid)
- `batch-size`: The batch size of the dataset

## IV. Result and Comparision

*15 layers:*

- Training results using Plain Networks:

```python
Epoch 9/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1121 - accuracy: 0.9702 - val_loss: 0.1607 - val_accuracy: 0.9586
Epoch 10/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1020 - accuracy: 0.9728 - val_loss: 0.1402 - val_accuracy: 0.9633

```

- Training results using Highway Networks:

```python
# t_bias=-2.0,acti_h = tf.nn.relu, acti_t = tf.nn.sigmoid
Epoch 9/10
1563/1563 [==============================] - 9s 6ms/step - loss: 0.2483 - accuracy: 0.9289 - val_loss: 0.2329 - val_accuracy: 0.9341
Epoch 10/10
1563/1563 [==============================] - 9s 6ms/step - loss: 0.2346 - accuracy: 0.9311 - val_loss: 0.2234 - val_accuracy: 0.9350

```

*50 layers:*

- Training results using Plain Networks:

```python
Epoch 9/10
1875/1875 [==============================] - 10s 6ms/step - loss: 2.3013 - accuracy: 0.1124 - val_loss: 2.3010 - val_accuracy: 0.1135
Epoch 10/10
1875/1875 [==============================] - 10s 5ms/step - loss: 2.3013 - accuracy: 0.1124 - val_loss: 2.3012 - val_accuracy: 0.1135

```

- Training results using Highway Networks:

```python
# t_bias=-3.0,acti_h = tf.nn.relu, acti_t = tf.nn.sigmoid
Epoch 9/10
1563/1563 [==============================] - 22s 14ms/step - loss: 0.2756 - accuracy: 0.9191 - val_loss: 0.2584 - val_accuracy: 0.9273
Epoch 10/10
1563/1563 [==============================] - 22s 14ms/step - loss: 0.2595 - accuracy: 0.9229 - val_loss: 0.2475 - val_accuracy: 0.9289

```
*1000 layers:*

- Training results using Highway Networks:

```python
# t_bias=-20.0,acti_h = tf.nn.relu, acti_t = tf.nn.sigmoid
Epoch 9/10
1563/1563 [==============================] - 279s 179ms/step - loss: 0.2877 - acc: 0.9191 - val_loss: 0.2822 - val_acc: 0.9207
Epoch 10/10
1563/1563 [==============================] - 273s 175ms/step - loss: 0.2849 - acc: 0.9205 - val_loss: 0.2801 - val_acc: 0.9218

```

To train a very deep neural network, you should start with high transform gate biases _**(more negative)**_ since it is easier to learn to overcome carrying than without carry gates (which is just a plain network).

By using the gating function, highway networks can optimize the depth directly. Specifically, the highway networks achieve comparable performance with the plain networks when the networks include only 15 layers **but** _**significantly outperform**_ (much higher accuracy) the plain networks when the networks go deeper to more than 50 layers.

The last result shows that highway networks can train up to 1000 layers with similar high accuracy as 50 layers. 
