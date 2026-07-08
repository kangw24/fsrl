科学问题：如何使用计算建模的方法复现人的行为学结果

检查：

已经使用 git clone 将 https://github.com/VeriTas-arch/fsrl 克隆，本地创建新分支，或者 fork 到自己账号下再进行克隆

难以本地训练的同学，注册好服务器（例如 https://www.autodl.com/home、https://featurize.cn/）账号，并配置好服务器中的环境，尝试跑通一次 simple_neo.py

代码理解：

阅读 simple_neo.py，了解`RetroModulRNN` 的网络架构

阅读 simple_neo.py，了解代码参数类 `TrainConfig` 中的每个参数含义，首先重点关注与网络尺寸、训练流程相关的参数，例如隐藏层大小、episode 个数

阅读 simple_neo.py，了解 `build_step_inputs` 、`sample_trial_pair` 函数，理解 input 拼接、训练集和测试集采样的范式

阅读 simple_neo.py，了解`run_episode` 如何构造并完成一个 episode

可做任务：

将行为学文章的 few-shot 任务 pair 的设计迁移到 simple_neo.py 中，更改`sample_trial_pair` 函数

更换网络尺寸（200 → 100）、episode（30000 → 100）等参数，跑通 simple_neo.py

可以尝试白板模型仿照 NN 文章先做 meta-learning 看结果

可以尝试先用 NN 文章的模型结果，将行为学文章的分析方法转变成代码，分析已有模型结果是否直接能解释行为学文章的结果

可以尝试在 NN 文章的模型基础上做 testing 或者 post-training

参数扫描

作业布置：

7.6：个人作业：

书写一份 proposal，发送到群里（proposal 要求已于群内公布）

更改`sample_trial_pair` 函数，将分支 push 到远端

完成 simple_neo.py 的训练