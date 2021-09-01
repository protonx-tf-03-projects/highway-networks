# Highway Network

Library Logo <--- **FIXME**

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Implementation for paper **Highway Network (2015)**. Give us a star if you like this repo. 

**Highway Networks** solve the optimization problem in a very deep neural network by using a gating function to adaptively transform or bypass the signal so that the network can go deeper to 1000 layers, much more than 10 layers in plain networks.

**Model architecture:**

<p align="center">
    <img src='https://github.com/protonx-tf-03-projects/highway-networks/blob/readme/Highway-Networks.png' width=800 class="center">
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

- The dataset used for Highway Networks is the MNIST database of handwritten digits, available from this page. It has a training set of 60000 examples, and a test set of 10000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
  
## III. Training Process

Training script:
```python
** FIX ME **
python train.py --epochs ${epochs} --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```
Example:

```python
** FIX ME **
!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --patch-size 5 --image-size 150 --lr 0.0001 --epochs 200 --num-heads 12 

```

There are some important arguments for the script you should consider when running it:

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

*10 layers:*
Training results using Plain Networks:
```
Epoch 8/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0243 - accuracy: 0.9921 - val_loss: 0.0796 - val_accuracy: 0.9770
Epoch 9/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0196 - accuracy: 0.9938 - val_loss: 0.0830 - val_accuracy: 0.9776
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0165 - accuracy: 0.9951 - val_loss: 0.0835 - val_accuracy: 0.9772
```
Training results using Highway Networks:
```
Epoch 8/10
1875/1875 [==============================] - 8s - loss: 0.0972 - accuracy: 0.9685 - val_loss: 0.1217 - val_accuracy: 0.9618
Epoch 9/10
1875/1875 [==============================] - 8s - loss: 0.0895 - accuracy: 0.9721 - val_loss: 0.1110 - val_accuracy: 0.9665
Epoch 10/10
1875/1875 [==============================] - 8s - loss: 0.0828 - accuracy: 0.9741 - val_loss: 0.1324 - val_accuracy: 0.9618
```

*19 layers:*
Training results using Plain Networks:
```
Epoch 8/10
1875/1875 [==============================] - 11s 6ms/step - loss: 0.1704 - accuracy: 0.9544 - val_loss: 0.1935 - val_accuracy: 0.9465
Epoch 9/10
1875/1875 [==============================] - 11s 6ms/step - loss: 0.1496 - accuracy: 0.9600 - val_loss: 0.1635 - val_accuracy: 0.9552
Epoch 10/10
1875/1875 [==============================] - 11s 6ms/step - loss: 0.1312 - accuracy: 0.9644 - val_loss: 0.1633 - val_accuracy: 0.9594
```
Training results using Highway Networks:
```
Epoch 8/10
1875/1875 [==============================] - 14s - loss: 0.1229 - accuracy: 0.9642 - val_loss: 0.1430 - val_accuracy: 0.9599
Epoch 9/10
1875/1875 [==============================] - 14s - loss: 0.1120 - accuracy: 0.9670 - val_loss: 0.1482 - val_accuracy: 0.9556
Epoch 10/10
1875/1875 [==============================] - 14s - loss: 0.1038 - accuracy: 0.9693 - val_loss: 0.1376 - val_accuracy: 0.9614
```

*29 layers:*
Training results using Plain Networks:
```
Epoch 8/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.8255 - accuracy: 0.7328 - val_loss: 0.5786 - val_accuracy: 0.8036
Epoch 9/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.6687 - accuracy: 0.8119 - val_loss: 0.4737 - val_accuracy: 0.8819
Epoch 10/10
1875/1875 [==============================] - 14s 7ms/step - loss: 0.5249 - accuracy: 0.8508 - val_loss: 0.4721 - val_accuracy: 0.8632
```
Training results using Highway Networks:
```
Epoch 8/10
1875/1875 [==============================] - 19s - loss: 0.1425 - accuracy: 0.9596 - val_loss: 0.1722 - val_accuracy: 0.9535
Epoch 9/10
1875/1875 [==============================] - 19s - loss: 0.1298 - accuracy: 0.9633 - val_loss: 0.1287 - val_accuracy: 0.9647
Epoch 10/10
1875/1875 [==============================] - 19s - loss: 0.1159 - accuracy: 0.9668 - val_loss: 0.1168 - val_accuracy: 0.9668
```

By using the gating function, Highway can optimize the deep directly. In particular, the highway network can obtain higher accuracy than plain network from 19 layers.


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


