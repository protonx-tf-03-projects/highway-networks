# Highway Network

Library Logo <--- **FIXME**

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Implementation for paper **Highway Network (2015)**. Give us a star if you like this repo. 

**Highway Networks** solve the optimization problem in a very deep neural network by using a gating function to adaptively transform or bypass the signal so that the network can go deeper to 1000 layers, much more than 10 layers in plain networks.

**Model architecture:**

<p align="center">
    <img src='https://github.com/protonx-tf-03-projects/highway-networks/blob/readme/Highway-Networks.png' width=200 class="center">
</p>


**Authors (Github & Email):**
- mrmessi2512 (team leader) – phungngbaolong@gmail.com
- quan030994 – tranquan030894@gmail.com
- tuvu247 – vungoctu136@gmail.com
- hatruong29 – maiha.th.92@gmail.com

**Advisors:**
- bangoc123 – protonxai@gmail.com

This library belongs to our project: Protonx-tf-03-projects where we implement AI papers and publish all source codes.

## I.  Set up environment
1.	Make sure you have installed Miniconda. If not yet, see the setup document [here]. 
2.	`cd` into `highway-networks` and use command line `conda env create -f environment.yml` to setup the environment
3.	Run conda environment using the command `conda activate highway-networks`

## II.  Set up your dataset

- Guide user how to download your data and set the data pipeline <--- **FIXME**
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)
The dataset used for Highway Networks is the MNIST database of handwritten digits, available from this page. It has a training set of 60000 examples, and a test set of 10000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
  
## III. Training Process


**FIXME**

Training script:


```python

python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```
**FIXME**

Example:

```python

!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --patch-size 5 --image-size 150 --lr 0.0001 --epochs 200 --num-heads 12 

``` 
**FIXME**

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- `num_classes`: the number of units (71) in plain network or blocks (50) in highway network
- `num_of_layers`: the number of layer in network
- `h`: the transform function followed by an activation function
- `t`: the transform gate which is a sigmoid function
- `x`: the input
- `t_bias`: the bias in the transform gate which can be initialized with an negative value such that the network is initially biased towards carry behavior.
- `acti_h`: the activation function following the transform function h (ReLU)
- `acti_t`: the activation function following the transform gate t (Tanh)
- `batch-size`: The batch size of the dataset


## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

**FIXME**

Training results using Plain Networks:
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Training results using Highway Networks:

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

By using the gating function, Highway can optimize the deep directly. In particular, the highway network can obtain highest accuracy with 19 layers.


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


