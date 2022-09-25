# 方法1思路
1. 去掉adaptive model的非线性操作，只保留线性操作
    + 在notears上
        + 10nodes，ER2和ER4是work的
        + 20nodes，ER2是work的，但是ER4不work
    + golem上有待验证

# 方法2思路
1. 因为只有一层linear，所以初始化是非常关键的，否则容易收敛到不是我们想要的值区间
    + 采用稀疏矩阵初始化
      + notears 20 80: 