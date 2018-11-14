# Snake-Bot |  sb_2 - deep neural network

```bash
Input layer (5 features)
1: left-obstacle ⋹ [0 for no, 1 for yes]
2: front-obstacle ⋹ [0 for no, 1 for yes]
3: right-obstacle ⋹ [0 for no, 1 for yes]
4: angle of the food relative to the snakehead ⋹ [-1 for -180° .. 1 for 180°]
5: distance between food and snakehead ⋹ [0 for very close .. 1 for very far]
--------------------------------------------
               Hidden layers
--------------------------------------------
Output layer (softmax => distribute the probability among n class, which sum to 1)
1: move left 
2: move straight 
3: move right
```

The sample video of the actual gameplay is [here](https://youtu.be/CmruOStJP5c).

### Investigate the changes with TensorBoard

```bash
$ tensorboard --logdir files/training_logs
```