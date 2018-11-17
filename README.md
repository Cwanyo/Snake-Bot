# Snake-Bot |  sb_3 - convolutional neural network

```bash
Input layer 
1: board state [22 x 22 x 1]

where the values in each pixel are:
    wall and snake’s body = -10
    freespace = 0
    snake’s head = 5
    food = 10
--------------------------------------------
               Hidden layers
--------------------------------------------
Output layer (softmax => distribute the probability among n class, which sum to 1)
1: move left - class 0
2: move up - class 1
3: move right - class 2
4: move down - class 3

```

The sample video of the actual gameplay is [here](https://youtu.be/qPK-ZB_fkys).

### Investigate the changes with TensorBoard

```bash
$ tensorboard --logdir files/training_logs
```