# Snake-Bot |  sb_1 - deep neural network

```bash
Input layer (5 features)
1: left-obstacle ⋹ [0 for no, 1 for yes]
2: front-obstacle ⋹ [0 for no, 1 for yes]
3: right-obstacle ⋹ [0 for no, 1 for yes]
4: angle of the food relative to the snakehead ⋹ [-1 for -180° .. 1 for 180°]
5: move direction ⋹ [-1 for left, 0 for straight, 1 for right]
--------------------------------------------
               Hidden layers
--------------------------------------------
Output layer (softmax => distribute the probability among n class, which sum to 1)
1: dead
2: alive but went to wrong direction
3: alive and went to right direction
```

The sample video of the actual gameplay is [here](https://youtu.be/GUVog0M-OjE).

### Investigate the changes with TensorBoard

```bash
$ tensorboard --logdir files/training_logs
```
