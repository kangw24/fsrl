# Constructive Global Rank 3.2

本分支只包含 Constructive Global Rank 3.2（CGR 3.2）的可运行实现与测试。

CGR 3.2 用一个可检查的 episode 内状态模拟从局部比较到整体排序的形成过程。模型在学习阶段逐条接收“哪个物品更高”及可见的相对高度差；每次呈现先经过概率编码，再用递归最小二乘（RLS）更新一维潜在排序轴，并立刻重建当前的全局顺序。进入测验阶段后，内部状态冻结，查询只读取最终顺序，不接收答案、奖励或反馈。

## 实验流程

- 8 个抽象物品。
- 学习阶段使用 8 个固定关系，每个关系重复 4 次，共 32 个 trial；每个 block 内随机打乱顺序并随机交换左右位置。
- 每次学习呈现的模型可见输入为物品身份、方向和归一化高度差。
- 测验阶段覆盖全部 28 个无序物品对，重复 10 个 block。
- 查询函数返回左侧物品更高的概率，而且不会修改 episode 状态。

## 模型机制

一次 episode 初始化时抽取物品锚点

```text
a_i ~ Normal(0, sigma_a^2),  sigma_a = 0.30
```

第 `t` 次呈现以 `p_enc = 0.45` 独立编码。编码事件由 episode 种子、物品对和该关系的曝光次数共同确定，因此调整不同关系的呈现顺序会改变中间轨迹，但不会把随机数分配差异误当作顺序效应。

若该呈现被编码，模型用观测向量 `x`（高物品为 `+1`、低物品为 `-1`）和可见高度差 `y` 做 RLS 更新：

```text
k = P x / (1 + x^T P x)
s <- s + k (y - x^T s)
P <- P - k x^T P
```

随后以 `s + a` 为偏好分数，对已经记住的方向关系做优先拓扑排序，得到当前的 provisional global order。这个顺序可被后续学习证据修正；实现没有把它解释为不可逆的心理承诺。

测验时，顺序位置先映射到 `[0, 1]`，再通过 sigmoid 得到选择概率：

```text
P(left is higher) = sigmoid(beta * (value_left - value_right)),  beta = 12
```

## 代码结构

```text
cgr_v3_2/model.py       在线状态、RLS 更新、全局序重建与冻结读取
cgr_v3_2/task.py        32 次学习呈现与 280 次无反馈查询的任务生成
tests/test_cgr_v3_2.py  输入边界、在线轨迹、只读查询与数值一致性测试
```

## 运行

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

最小调用示例：

```python
import numpy as np

from cgr_v3_2 import CGR32, build_subject_tasks

task = build_subject_tasks(
    n_subjects=1,
    rng=np.random.default_rng(7),
)[0]

model = CGR32(n_items=8)
run, state = model.run_subject_with_state(task, np.random.default_rng(21))

print(run.order_strong_to_weak)
print(state.order_trajectory)
```

## 解释边界

CGR 3.2 是一个功能层面的认知候选：模块分别对应有限编码、证据整合、整体顺序构造和无反馈读取，并允许逐 trial 检查内部变化。它不是 RNN，也不能仅凭行为拟合证明人类大脑使用 RLS。当前实现适合检验“这些计算角色能否共同产生目标行为”，而不是把工程模块直接等同于已确认的神经机制。
