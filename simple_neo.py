"""
Readable training script for the plastic RNN transitive-inference task.

This file keeps the learning rule, task structure, and saved model format of
``simple.py`` while organizing the code into small functions for learning.
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADDINPUT = 4
NUMRESPONSESTEP = 1


def log(message):
    print(message, flush=True)


@dataclass
class TrainConfig:
    rngseed: int = -1
    rew: float = 1.0
    wp: float = 0.0
    bent: float = 0.1
    blossv: float = 0.1
    gr: float = 0.9
    hs: int = 200
    bs: int = 32
    gc: float = 2.0
    eps: float = 1e-6
  # nbiter: int = 30000  # 旧版短 episode 用的迭代数
    nbiter: int = 5000  # 新任务每 episode ≈1184 步，总计算量约为旧版 30000×120 的数倍
    save_every: int = 100
    pe: int = 101
    cs: int = 15
    # triallen: int = 4  # 旧版：所有 trial 统一步数
    learn_triallen: int = 2  # 学习阶段：展示约束 + 观察（无 Go / 无决策）
    test_triallen: int = 4  # 测试阶段：保留 Go 信号与决策步
    # nbtraintrials: int = 8   # 旧版：单轮学习 trial 数
    # nbtesttrials: int = 28   # 旧版：单轮测试 trial 数
    nb_learn_blocks: int = 4  # 学习 block 数（每 block 呈现完整监督集 S）
    nb_test_blocks: int = 10  # 测试 block 数（每 block 穷尽 Q 中 28 个 pair）
    supervision_size: int = 8  # 监督集 S 大小（非相邻约束数）
    testlmult: float = 3.0
    l2: float = 0.0
    lr: float = 1e-4
    lpw: float = 1e-4
    nbcues: int = 8  # Mermaid 默认：8 个元素；全序 0≻1≻…≻7（编号越小越强）
    analyze_on_save: bool = False  # checkpoint 时是否运行分析环节
    # 观察学习：随机左右呈现，使 teacher_da 有正有负（stim1 赢 → +1，stim2 赢 → -1）
    learn_shuffle_presentation: bool = True
    # 离线辅助监督（行为仍无反馈；仅用于 meta-train 反传）
    baux_learn: float = 0.5  # 学习阶段 pair 呈现步的 CE
    baux_test: float = 1.0  # 测试决策步的 CE

    @property
    def nbcuesrange(self):
        return range(self.nbcues, self.nbcues + 1)

    @property
    def n_learn_trials(self):
        """每个 episode 的学习 trial 总数 = block 数 × |S|。"""
        return self.nb_learn_blocks * self.supervision_size

    @property
    def n_test_trials(self):
        """每个 episode 的测试 trial 总数 = block 数 × C(n,2)。"""
        return self.nb_test_blocks * (self.nbcues * (self.nbcues - 1) // 2)

    @property
    def nbtrials(self):
        """每个 episode 的总 trial 数。"""
        return self.n_learn_trials + self.n_test_trials

    @property
    def eplen(self):
        """每个 episode 的总时间步数（学习步 + 测试步）。"""
        return (
            self.n_learn_trials * self.learn_triallen
            + self.n_test_trials * self.test_triallen
        )

    @property
    def triallen(self):
        """兼容旧接口：默认返回测试 trial 步数。"""
        return self.test_triallen

    @property
    def nbtraintrials(self):
        """兼容旧接口：返回学习 trial 总数。"""
        return self.n_learn_trials

    @property
    def nbtesttrials(self):
        """兼容旧接口：返回测试 trial 总数。"""
        return self.n_test_trials

    @property
    def nbstimbits(self):
        return 2 * self.cs + 1

    @property
    def outputsize(self):
        return 2

    @property
    def inputsize(self):
        return self.nbstimbits + ADDINPUT + self.outputsize

    def to_model_dict(self):
        """Return the dict shape expected by the original model code."""
        return {
            "rngseed": self.rngseed,
            "rew": self.rew,
            "wp": self.wp,
            "bent": self.bent,
            "blossv": self.blossv,
            "gr": self.gr,
            "hs": self.hs,
            "bs": self.bs,
            "gc": self.gc,
            "eps": self.eps,
            "nbiter": self.nbiter,
            "save_every": self.save_every,
            "pe": self.pe,
            "nbcuesrange": self.nbcuesrange,
            "cs": self.cs,
            "triallen": self.test_triallen,
            "learn_triallen": self.learn_triallen,
            "test_triallen": self.test_triallen,
            "nb_learn_blocks": self.nb_learn_blocks,
            "nb_test_blocks": self.nb_test_blocks,
            "supervision_size": self.supervision_size,
            "nbtraintrials": self.nbtraintrials,
            "nbtesttrials": self.nbtesttrials,
            "nbtrials": self.nbtrials,
            "eplen": self.eplen,
            "testlmult": self.testlmult,
            "l2": self.l2,
            "lr": self.lr,
            "lpw": self.lpw,
            "outputsize": self.outputsize,
            "inputsize": self.inputsize,
        }


class RetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity.

    ``et`` is the Hebbian eligibility trace. ``pw`` is the within-episode
    plastic recurrent weight matrix. Only ``pw`` changes during an episode.
    """

    def __init__(self, config):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in config:
                raise KeyError("Must provide missing key in config: " + paramname)

        nbda = 2
        self.GG = config
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(config["inputsize"], config["hs"]).to(DEVICE)
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(config["hs"]))
                * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)
            ).to(DEVICE),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (0.01 * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)).to(DEVICE),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.h2DA = torch.nn.Linear(config["hs"], nbda).to(DEVICE)
        self.h2o = torch.nn.Linear(config["hs"], config["outputsize"]).to(DEVICE)
        self.h2v = torch.nn.Linear(config["hs"], 1).to(DEVICE)

    def forward(self, inputs, hidden, et, pw, teacher_da=None):
        batch_size = inputs.shape[0]
        hidden_size = self.GG["hs"]
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size

        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                (self.w + torch.mul(self.alpha, pw)),
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)

        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)

        daout2 = torch.tanh(self.h2DA(hactiv))
        # 学习阶段可注入教师多巴胺信号，用于观察学习（无需被试显式选择）
        if teacher_da is not None:
            daout = teacher_da
        else:
            daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        pw = pw + daout.view(batch_size, 1, 1) * et
        torch.clip_(pw, min=-50.0, max=50.0)

        deltaet = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, et, pw

    def initialZeroET(self, batch_size):
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], requires_grad=False).to(DEVICE)


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    loss_objective: float
    test_reward_mean: float
    nbtesttrials: int
    test_perf: float | None
    test_perf_adjacent: float | None
    test_perf_nonadjacent: float | None
    final_pw: torch.Tensor


@dataclass
class TestResponse:
    """单次测试查询的响应记录（用于分析阶段）。"""

    block_id: int
    pair: tuple[int, int]  # canonical (i, j), i < j
    action: int  # 0 或 1
    prefers_i_over_j: bool  # 是否偏好第一个 cue（编号较小者）
    correct: bool
    batch_index: int


@dataclass
class EpisodeRecord:
    """单次 eval episode 的完整记录，供 analysis.py 使用。"""

    nbcues: int
    supervision_set: list
    query_set: list
    test_responses: list[TestResponse] = field(default_factory=list)
    true_rank: list[int] = field(default_factory=list)
    test_perf: float | None = None
    test_perf_adjacent: float | None = None
    test_perf_nonadjacent: float | None = None


def set_seed(seed):
    if seed < 0:
        log("[setup] No random seed.")
        return
    log(f"[setup] Setting random seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_cue_data(config, nbcues):
    """Generate unique random binary cue vectors for each batch element."""
    cue_data = []
    for batch_index in range(config.bs):
        cue_data.append([])
        for cue_index in range(nbcues):
            candidate = sample_unique_cue(config, cue_data[batch_index], cue_index)
            cue_data[batch_index].append(candidate)
    return cue_data


def sample_unique_cue(config, existing_cues, cue_index):
    attempts = 0
    while True:
        attempts += 1
        if attempts > 10000:
            raise ValueError("Could not generate a full list of different cues")

        candidate = np.random.randint(2, size=config.cs) * 2 - 1
        is_too_similar = False
        for previous_index in range(cue_index):
            if np.mean(existing_cues[previous_index] == candidate) > 0.66:
                is_too_similar = True
                break
        if not is_too_similar:
            return candidate


def build_supervision_set(nbcues, size=8):
    """构建局部监督集合 S：非相邻 pairwise constraints，含秩差 Δᵢⱼ。"""
    non_adjacent = [
        (i, j, abs(i - j))
        for i in range(nbcues)
        for j in range(i + 1, nbcues)
        if abs(i - j) > 1
    ]
    non_adjacent.sort(key=lambda item: (item[2], item[0], item[1]))
    if size < len(non_adjacent):
        return non_adjacent[:size]
    return non_adjacent


def build_query_set(nbcues):
    """构建全集比较查询 Q：所有 C(n,2) 个 unordered pairs。"""
    return [(i, j) for i in range(nbcues) for j in range(i + 1, nbcues)]


def shuffle_schedule(schedule):
    """每个 block 开始前打乱呈现顺序。"""
    shuffled = list(schedule)
    np.random.shuffle(shuffled)
    return shuffled


def sample_trial_pair(nbcues, is_train_trial):
    """【已弃用】旧版随机采样 trial pair，现由 block schedule 取代。"""
    cue_pair = list(np.random.choice(range(nbcues), 2, replace=False))
    if is_train_trial:
        # 旧版：训练阶段强制相邻
        # while abs(cue_pair[0] - cue_pair[1]) > 1:
        # 新版：训练阶段强制非相邻（已由 build_supervision_set 保证）
        while abs(cue_pair[0] - cue_pair[1]) <= 1:
            cue_pair = list(np.random.choice(range(nbcues), 2, replace=False))
    return cue_pair


def build_step_inputs(
    config,
    nbcues,
    cue_data,
    cues,
    reward,
    previous_actions,
    numstep,
    numstep_ep,
    phase,
    delta=0,
):
    """按 trial 阶段构造网络输入；学习阶段无 Go 信号与动作反馈。"""
    inputs = np.zeros((config.bs, config.inputsize), dtype="float32")

    for batch_index in range(config.bs):
        cue = cues[batch_index][numstep]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.nbstimbits - 1] = np.concatenate(
                (cue_data[batch_index][cue[0]][:], cue_data[batch_index][cue[1]][:])
            )
        # 仅测试阶段在决策步前给出 Go 信号
        elif cue == nbcues and phase == "test":
            inputs[batch_index, config.nbstimbits - 1] = 1
        # 旧版：学习/测试统一给 Go 信号
        # elif cue == nbcues:
        #     inputs[batch_index, config.nbstimbits - 1] = 1

        inputs[batch_index, config.nbstimbits + 0] = 1.0
        inputs[batch_index, config.nbstimbits + 1] = numstep_ep / config.eplen
        inputs[batch_index, config.nbstimbits + 2] = reward[batch_index]
        # 秩差 Δᵢⱼ 写入时间通道旁的第 4 个附加输入（归一化）
        if config.nbcues > 1 and delta > 0:
            inputs[batch_index, config.nbstimbits + 3] = delta / (config.nbcues - 1)

        # 仅测试阶段回传上一步动作
        if phase == "test" and numstep == NUMRESPONSESTEP + 1:
            inputs[
                batch_index,
                config.nbstimbits + ADDINPUT + previous_actions[batch_index],
            ] = 1
        # 旧版：学习/测试统一回传动作
        # if numstep == NUMRESPONSESTEP + 1:
        #     inputs[
        #         batch_index,
        #         config.nbstimbits + ADDINPUT + previous_actions[batch_index],
        #     ] = 1

    return torch.from_numpy(inputs).detach().to(DEVICE)


def stronger_cue_index(i: int, j: int) -> int:
    """真实全序：编号越小越强（0 最强）。"""
    return min(i, j)


def weaker_cue_index(i: int, j: int) -> int:
    return max(i, j)


def default_true_rank(nbcues: int) -> list[int]:
    """从强到弱的 cue 编号列表（与 stronger_cue_index 一致）。"""
    return list(range(nbcues))


def prepare_trial(config, nbcues, cue_pair, phase):
    """根据 block schedule 给定 pair 构造 trial，不再随机采样。"""
    cues = []
    cue_pairs = []
    correct_order = np.zeros(config.bs)
    adjacent = np.zeros(config.bs)
    teacher_signal = np.zeros(config.bs, dtype=np.float32)

    if len(cue_pair) == 3:
        i, j, delta = cue_pair[0], cue_pair[1], cue_pair[2]
    else:
        i, j = cue_pair[0], cue_pair[1]
        delta = abs(i - j)

    stronger = stronger_cue_index(i, j)
    weaker = weaker_cue_index(i, j)

    # 测试：固定呈现 (强, 弱) = (小编号, 大编号)，如 [0, 7]
    # 学习：可随机交换左右，使 teacher_da 随「赢家在哪侧」变化
    if phase == "learn" and config.learn_shuffle_presentation:
        if np.random.random() < 0.5:
            display_pair = [weaker, stronger]
        else:
            display_pair = [stronger, weaker]
    else:
        display_pair = [stronger, weaker]

    for batch_index in range(config.bs):
        stim1_is_stronger = display_pair[0] == stronger
        # 1 → 应选 stim1（action 1）；0 → 应选 stim2（action 0）
        correct_order[batch_index] = 1 if stim1_is_stronger else 0
        adjacent[batch_index] = 1 if abs(i - j) == 1 else 0
        cue_pairs.append(list(display_pair))
        teacher_signal[batch_index] = 1.0 if stim1_is_stronger else -1.0
        if phase == "learn":
            cues.append([display_pair, -1, -1, -1])
        else:
            cues.append([display_pair, nbcues, -1, -1])

    return cues, cue_pairs, correct_order, adjacent, delta, teacher_signal


def compute_trial_loss(
    config, trial_rewards, trial_values, trial_logprobs, bent, triallen, aux_loss=None
):
    """
    单个 trial 内的损失（bent + 价值预测 + 策略梯度占位 + 可选辅助 CE）。
    配合 trial 末 pw.detach()，避免 1184 步整图反传导致 OOM。
    """
    loss = bent
    lossv = 0
    bootstrap_return = torch.zeros(config.bs, device=DEVICE)
    for step_idx in reversed(range(triallen)):
        bootstrap_return = config.gr * bootstrap_return + torch.from_numpy(
            trial_rewards[step_idx]
        ).to(DEVICE)
        advantage = bootstrap_return - trial_values[step_idx][:, 0]
        lossv = lossv + advantage.pow(2).sum() / config.bs
        loss_multiplier = 0.0  # 行为无 reward 反馈；策略梯度仍关闭
        loss = (
            loss
            - loss_multiplier
            * (trial_logprobs[step_idx] * advantage.detach()).sum()
            / config.bs
        )
    loss = loss + config.blossv * lossv
    if aux_loss is not None:
        loss = loss + aux_loss
    return loss / triallen


def run_single_trial(
    config,
    net,
    nbcues,
    cue_data,
    scheduled_pair,
    phase,
    global_trial_idx,
    hidden,
    et,
    pw,
    numstep_ep,
    correct_thisep,
    istest_thisep,
    test_counters,
    print_trace=False,
    test_responses=None,
    block_id=None,
    compute_loss=False,
):
    """执行单个 trial（学习或测试），并记录逐步状态。"""
    triallen = config.learn_triallen if phase == "learn" else config.test_triallen

    # 每个 trial 重置隐状态与资格迹；可塑权重 pw 跨 trial 保留（值保留，图在 trial 末截断）
    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)

    trial_rewards = []
    trial_values = []
    trial_logprobs = []
    bent = torch.tensor(0.0, device=DEVICE)
    trial_aux_loss = torch.tensor(0.0, device=DEVICE)

    cues, cue_pairs, correct_order, adjacent, delta, teacher_signal = prepare_trial(
        config, nbcues, scheduled_pair, phase
    )
    istest_thisep[:, global_trial_idx] = 1 if phase == "test" else 0

    correct_answer = np.zeros(config.bs)
    previous_actions = np.zeros(config.bs, dtype=np.int32)
    reward = np.zeros(config.bs, dtype="float32")

    for numstep in range(triallen):
        inputs = build_step_inputs(
            config,
            nbcues,
            cue_data,
            cues,
            reward,
            previous_actions,
            numstep,
            numstep_ep,
            phase,
            delta=delta,
        )

        teacher_da = None
        if phase == "learn":
            # 观察学习：stim1 为更强项 → +1，否则 -1（配合随机左右呈现）
            teacher_da = torch.from_numpy(teacher_signal).to(DEVICE).view(config.bs, 1)

        y_raw, value, daout, hidden, et, pw = net(
            inputs, hidden, et, pw, teacher_da=teacher_da
        )

        if phase == "test":
            y = torch.softmax(y_raw, dim=1)
            distrib = torch.distributions.Categorical(y)
            actions = distrib.sample()
            trial_logprobs.append(distrib.log_prob(actions))
            previous_actions = actions.detach().cpu().numpy()
        else:
            trial_logprobs.append(torch.zeros(config.bs, device=DEVICE))
            previous_actions = np.zeros(config.bs, dtype=np.int32)
            y = torch.softmax(y_raw, dim=1)

        if compute_loss:
            target = torch.from_numpy(correct_order.astype(np.int64)).to(DEVICE)
            if phase == "learn" and numstep == 0 and config.baux_learn > 0:
                # pair 呈现步：离线预测更强的一侧（action 1 = stim1）
                trial_aux_loss = trial_aux_loss + config.baux_learn * nn.functional.cross_entropy(
                    y_raw, target
                )
            if phase == "test" and numstep == NUMRESPONSESTEP and config.baux_test > 0:
                # 决策步：离线 CE，行为仍无 reward 反馈
                trial_aux_loss = trial_aux_loss + config.baux_test * nn.functional.cross_entropy(
                    y_raw, target
                )

        if print_trace:
            log_trace(
                config,
                global_trial_idx,
                numstep,
                inputs,
                y,
                previous_actions,
                correct_order,
                reward,
                daout,
                cues,
                phase=phase,
            )

        reward = np.zeros(config.bs, dtype="float32")
        for batch_index in range(config.bs):
            if phase == "test" and numstep == NUMRESPONSESTEP:
                correct_answer[batch_index] = 1
                chose_stim1 = previous_actions[batch_index] == 1
                if (correct_order[batch_index] and chose_stim1) or (
                    (not correct_order[batch_index]) and not chose_stim1
                ):
                    # 测试无反馈：记录正确性，但不给 reward
                    pass
                    # 旧版：测试阶段仍给奖惩信号
                    # reward[batch_index] += config.rew
                else:
                    correct_answer[batch_index] = 0
                    # reward[batch_index] -= config.rew
                correct_thisep[batch_index, global_trial_idx] = correct_answer[batch_index]

                # 分析环节：记录测试响应方向（无反馈，仅离线采集）
                if test_responses is not None:
                    pair = cue_pairs[batch_index]
                    i, j = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                    prefers_i_over_j = previous_actions[batch_index] == 1
                    if pair[0] != i:
                        prefers_i_over_j = not prefers_i_over_j
                    test_responses.append(
                        TestResponse(
                            block_id=block_id if block_id is not None else -1,
                            pair=(i, j),
                            action=int(previous_actions[batch_index]),
                            prefers_i_over_j=bool(prefers_i_over_j),
                            correct=bool(correct_answer[batch_index]),
                            batch_index=batch_index,
                        )
                    )
            elif phase == "learn" and numstep == NUMRESPONSESTEP:
                # 旧版：学习阶段根据选择给 reward
                # correct_answer[batch_index] = 1
                # chose_item_1 = previous_actions[batch_index] == 1
                # if (correct_order[batch_index] and chose_item_1) or (
                #     (not correct_order[batch_index]) and not chose_item_1
                # ):
                #     reward[batch_index] += config.rew
                # else:
                #     reward[batch_index] -= config.rew
                #     correct_answer[batch_index] = 0
                pass

        trial_rewards.append(reward)
        trial_values.append(value)
        bent = bent + config.bent * y.pow(2).sum() / config.bs
        numstep_ep += 1

    trial_loss = None
    if compute_loss:
        trial_loss = compute_trial_loss(
            config,
            trial_rewards,
            trial_values,
            trial_logprobs,
            bent,
            triallen,
            aux_loss=trial_aux_loss,
        )
        trial_loss = trial_loss + config.lpw * torch.mean(pw**2) / config.nbtrials

    if phase == "test":
        test_counters["nbtesttrials"] += config.bs
        test_counters["nbtesttrials_correct"] += int(np.sum(correct_answer))
        test_counters["nbtesttrials_adjacent"] += int(np.sum(adjacent))
        test_counters["nbtesttrials_adjacent_correct"] += int(
            np.sum(adjacent * correct_answer)
        )
        test_counters["nbtesttrials_nonadjacent"] += int(np.sum(1 - adjacent))
        test_counters["nbtesttrials_nonadjacent_correct"] += int(
            np.sum((1 - adjacent) * correct_answer)
        )

    # 截断 BPTT：保留 pw 数值，切断跨 trial 的计算图，防止 1184 步整图 OOM
    pw = pw.detach()

    return hidden, et, pw, numstep_ep, trial_loss


def _run_episode_blocks(
    config,
    net,
    nbcues,
    print_trace=False,
    collect_responses=False,
    training=False,
):
    """
    执行 episode 的学习 + 测试 block 循环（训练与 eval 共用）。
    training=True 时按 trial 反传（截断 BPTT），避免整 episode 图导致 OOM。
    """
    hidden = net.initialZeroState(config.bs)
    et = net.initialZeroET(config.bs)
    pw = net.initialZeroPlasticWeights(config.bs)
    cue_data = generate_cue_data(config, nbcues)

    test_responses = [] if collect_responses else None
    correct_thisep = np.zeros((config.bs, config.nbtrials))
    istest_thisep = np.zeros((config.bs, config.nbtrials))

    test_counters = {
        "nbtesttrials": 0,
        "nbtesttrials_correct": 0,
        "nbtesttrials_adjacent": 0,
        "nbtesttrials_adjacent_correct": 0,
        "nbtesttrials_nonadjacent": 0,
        "nbtesttrials_nonadjacent_correct": 0,
    }

    loss_sum = 0.0

    blank_inputs = torch.zeros(config.bs, config.inputsize, requires_grad=False).to(
        DEVICE
    )
    for _ in range(2):
        _, _, _, hidden, et, pw = net(blank_inputs, hidden, et, pw)

    supervision_set = build_supervision_set(nbcues, config.supervision_size)
    query_set = build_query_set(nbcues)
    global_trial_idx = 0
    numstep_ep = 0

    def _run_scheduled_trial(scheduled_pair, phase, block_id):
        nonlocal hidden, et, pw, numstep_ep, loss_sum
        hidden, et, pw, numstep_ep, trial_loss = run_single_trial(
            config,
            net,
            nbcues,
            cue_data,
            scheduled_pair,
            phase=phase,
            global_trial_idx=global_trial_idx,
            hidden=hidden,
            et=et,
            pw=pw,
            numstep_ep=numstep_ep,
            correct_thisep=correct_thisep,
            istest_thisep=istest_thisep,
            test_counters=test_counters,
            print_trace=print_trace,
            test_responses=test_responses,
            block_id=block_id,
            compute_loss=training,
        )
        if training and trial_loss is not None:
            scaled = trial_loss / config.nbtrials
            scaled.backward()
            loss_sum += float(trial_loss.detach())

    for learn_block in range(config.nb_learn_blocks):
        learn_schedule = shuffle_schedule(supervision_set)
        for scheduled_pair in learn_schedule:
            _run_scheduled_trial(scheduled_pair, "learn", None)
            global_trial_idx += 1

    for test_block in range(config.nb_test_blocks):
        test_schedule = shuffle_schedule(query_set)
        for scheduled_pair in test_schedule:
            _run_scheduled_trial(scheduled_pair, "test", test_block)
            global_trial_idx += 1

    return {
        "loss_sum": loss_sum,
        "test_counters": test_counters,
        "pw": pw,
        "supervision_set": supervision_set,
        "query_set": query_set,
        "test_responses": test_responses or [],
    }


def run_episode(config, net, nbcues, print_trace=False):
    """运行完整 episode：学习 blocks + 测试 blocks；按 trial 反传，返回损失统计。"""
    state = _run_episode_blocks(
        config,
        net,
        nbcues,
        print_trace=print_trace,
        collect_responses=False,
        training=True,
    )

    pw = state["pw"]
    test_counters = state["test_counters"]
    loss_sum = state["loss_sum"]
    loss_value = loss_sum / max(config.nbtrials, 1)
    loss_tensor = torch.tensor(loss_value, device=DEVICE)

    nbtesttrials = test_counters["nbtesttrials"]
    nbtesttrials_correct = test_counters["nbtesttrials_correct"]
    nbtesttrials_adjacent = test_counters["nbtesttrials_adjacent"]
    nbtesttrials_adjacent_correct = test_counters["nbtesttrials_adjacent_correct"]
    nbtesttrials_nonadjacent = test_counters["nbtesttrials_nonadjacent"]
    nbtesttrials_nonadjacent_correct = test_counters[
        "nbtesttrials_nonadjacent_correct"
    ]

    test_perf = None if nbtesttrials == 0 else nbtesttrials_correct / nbtesttrials
    test_perf_adjacent = None
    if nbtesttrials_adjacent > 0:
        test_perf_adjacent = nbtesttrials_adjacent_correct / nbtesttrials_adjacent
    test_perf_nonadjacent = None
    if nbtesttrials_nonadjacent > 0:
        test_perf_nonadjacent = (
            nbtesttrials_nonadjacent_correct / nbtesttrials_nonadjacent
        )

    return EpisodeStats(
        loss=loss_tensor,
        loss_value=loss_value,
        loss_objective=loss_value,
        test_reward_mean=0.0,
        nbtesttrials=nbtesttrials,
        test_perf=test_perf,
        test_perf_adjacent=test_perf_adjacent,
        test_perf_nonadjacent=test_perf_nonadjacent,
        final_pw=pw.detach(),
    )


def resolve_model_path(model_path=None):
    """解析模型权重路径（net.dat 或 net_active.dat）。"""
    if model_path:
        path = Path(model_path)
        return path if path.is_absolute() else ROOT_DIR / path
    for candidate in ("net.dat", "net_active.dat"):
        path = ROOT_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "未找到模型文件。请在项目根目录放置 net.dat 或 net_active.dat，"
        "或通过 --model-path 指定。"
    )


def load_model_state(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_network(config, model_path=None):
    """加载已训练网络（eval / 分析用）。"""
    path = resolve_model_path(model_path)
    log(f"[model] 从 {path} 加载参数")
    model_config = config.to_model_dict()
    net = RetroModulRNN(model_config)
    net.load_state_dict(load_model_state(path))
    net.eval()
    return net


def run_episode_eval(config, net, nbcues, print_trace=False):
    """
    运行 eval episode 并采集 EpisodeRecord（供分析环节使用）。
    不计算训练 loss，不反传梯度。
    """
    torch.set_grad_enabled(False)
    net.eval()
    state = _run_episode_blocks(
        config, net, nbcues, print_trace=print_trace, collect_responses=True
    )
    test_counters = state["test_counters"]
    nbtesttrials = test_counters["nbtesttrials"]
    nb_correct = test_counters["nbtesttrials_correct"]

    test_perf = None if nbtesttrials == 0 else nb_correct / nbtesttrials
    test_perf_adjacent = None
    if test_counters["nbtesttrials_adjacent"] > 0:
        test_perf_adjacent = (
            test_counters["nbtesttrials_adjacent_correct"]
            / test_counters["nbtesttrials_adjacent"]
        )
    test_perf_nonadjacent = None
    if test_counters["nbtesttrials_nonadjacent"] > 0:
        test_perf_nonadjacent = (
            test_counters["nbtesttrials_nonadjacent_correct"]
            / test_counters["nbtesttrials_nonadjacent"]
        )

    return EpisodeRecord(
        nbcues=nbcues,
        supervision_set=state["supervision_set"],
        query_set=state["query_set"],
        test_responses=state["test_responses"],
        true_rank=default_true_rank(nbcues),
        test_perf=test_perf,
        test_perf_adjacent=test_perf_adjacent,
        test_perf_nonadjacent=test_perf_nonadjacent,
    )


def log_trace(
    config,
    numtrial,
    numstep,
    inputs,
    y,
    actions,
    correct_order,
    reward,
    daout,
    cues,
    phase="test",
):
    log(
        "Phase {} Tr {} Step {} Cue1(0): {} Cue2(0): {} Other inputs: {}\n"
        " - Outputs(0): {} - action chosen(0): {} TrialLen: {} numstep {} "
        "TTHCC(0): {} Reward(prev): {} DAout: {} cues(0): {}".format(
            phase,
            numtrial,
            numstep,
            inputs[0, : config.cs].detach().cpu().numpy(),
            inputs[0, config.cs : 2 * config.cs].detach().cpu().numpy(),
            inputs[0, 2 * config.cs :].detach().cpu().numpy(),
            y.detach().cpu().numpy()[0, :],
            actions[0],
            config.test_triallen if phase == "test" else config.learn_triallen,
            numstep,
            correct_order[0],
            reward[0],
            float(daout[0].detach()),
            cues[0],
        )
    )


def print_episode_summary(config, episode_index, stats, start_time):
    elapsed = time.time() - start_time
    log(f"Episode {episode_index} ====")
    log(f"Time spent on last {config.pe} iters: {elapsed:.2f}s")
    log(f"Mean loss: {stats.loss_value:.6f}")
    log(
        "Test performance: {} | adjacent: {} | nonadjacent: {}".format(
            "N/A" if stats.test_perf is None else f"{stats.test_perf:.3f}",
            (
                "N/A"
                if stats.test_perf_adjacent is None
                else f"{stats.test_perf_adjacent:.3f}"
            ),
            (
                "N/A"
                if stats.test_perf_nonadjacent is None
                else f"{stats.test_perf_nonadjacent:.3f}"
            ),
        )
    )
    pw = stats.final_pw
    log(f"mean-abs pw: {float(torch.mean(torch.abs(pw))):.6f}")


def save_checkpoint(config, net, output_dir, test_rewards):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output_dir / ("netAE" + str(config.rngseed) + ".dat"))
    torch.save(net.state_dict(), output_dir / "net.dat")
    with open(output_dir / ("tAE" + str(config.rngseed) + ".txt"), "w") as thefile:
        for item in test_rewards[::10]:
            thefile.write(f"{item}\n")
    log(f"[save] Wrote checkpoint and test-reward log to {output_dir}")


def train(config, output_dir, trace_steps=False):
    model_config = config.to_model_dict()
    net = RetroModulRNN(model_config)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2
    )
    test_rewards = []

    log(f"[setup] Device: {DEVICE}")
    log(
        f"[setup] Batch size: {config.bs}; episodes: {config.nbiter}; "
        f"learn_blocks: {config.nb_learn_blocks}; test_blocks: {config.nb_test_blocks}; "
        f"baux_learn={config.baux_learn}; baux_test={config.baux_test}; "
        f"learn_shuffle={config.learn_shuffle_presentation}; "
        f"output: {output_dir}"
    )
    log(f"[setup] Parameter shapes: {[x.size() for x in net.parameters()]}")

    trace_start = time.time()
    episode_iter = tqdm(
        range(config.nbiter),
        desc="training episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for episode_index in episode_iter:
        should_print_summary = episode_index % config.pe == 0
        print_trace = trace_steps and should_print_summary
        nbcues = config.nbcues

        optimizer.zero_grad()
        stats = run_episode(config, net, nbcues=nbcues, print_trace=print_trace)
        # 梯度已在 run_episode 内按 trial 累积反传
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        if episode_index > 100:
            optimizer.step()

        test_rewards.append(stats.test_reward_mean)
        if should_print_summary:
            print_episode_summary(config, episode_index, stats, trace_start)
            trace_start = time.time()

        if episode_index % config.save_every == 0 and episode_index > 0:
            save_checkpoint(config, net, output_dir, test_rewards)
            if config.analyze_on_save:
                _run_analysis_checkpoint(config, net, output_dir / "analysis")

    return net


def _run_analysis_checkpoint(config, net, analysis_dir):
    """训练过程中 checkpoint 时运行分析（可选）。"""
    from analysis import run_full_analysis

    log("[analysis] checkpoint 触发分析环节...")
    record = run_episode_eval(config, net, nbcues=config.nbcues)
    run_full_analysis(record, analysis_dir)


def run_analyze_mode(config, model_path, analysis_dir):
    """仅运行 eval + 分析（--analyze 模式）。"""
    from analysis import run_full_analysis

    net = load_network(config, model_path)
    analysis_dir = Path(analysis_dir)
    log(
        f"[analyze] batch={config.bs}, learn_blocks={config.nb_learn_blocks}, "
        f"test_blocks={config.nb_test_blocks}, |S|={config.supervision_size}"
    )
    record = run_episode_eval(config, net, nbcues=config.nbcues)
    report = run_full_analysis(record, analysis_dir)
    log(f"[analyze] 主观排序: {report.subjective_rank}")
    return report


def parse_args():
    # Mermaid 默认任务参数（不传 CLI 时即生效）：
    #   nbcues=8, learn_blocks=4, test_blocks=10, supervision_size=8
    #   → 学习 4×8=32 trials，测试 10×28=280 trials
    parser = argparse.ArgumentParser(
        description=(
            "可塑性 RNN 传递推理元训练。"
            "默认参数对齐 Mermaid 任务：8 元素、4 学习 block、10 测试 block、|S|=8。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nbiter", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=101)
    parser.add_argument("--output-dir", default=str(ROOT_DIR))
    parser.add_argument("--nbcues", type=int, default=8, help="元素数量（Mermaid 默认 8）")
    parser.add_argument("--learn-blocks", type=int, default=4, help="学习 block 数（Mermaid 默认 4）")
    parser.add_argument("--test-blocks", type=int, default=10, help="测试 block 数（Mermaid 默认 10）")
    parser.add_argument(
        "--supervision-size", type=int, default=8, help="监督集 S 大小（Mermaid 默认 8）"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="仅运行 eval + 响应结构分析（不训练）",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="分析模式使用的模型权重路径（默认 net.dat 或 net_active.dat）",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="分析结果输出目录（默认 <output-dir>/analysis）",
    )
    parser.add_argument(
        "--analyze-on-save",
        action="store_true",
        help="训练时每次 save checkpoint 后运行分析",
    )
    parser.add_argument(
        "--no-learn-shuffle",
        action="store_true",
        help="学习阶段不随机交换左右呈现（关闭有符号 teacher_da）",
    )
    parser.add_argument(
        "--baux-learn",
        type=float,
        default=0.5,
        help="学习 pair 呈现步离线 CE 权重（meta-train）",
    )
    parser.add_argument(
        "--baux-test",
        type=float,
        default=1.0,
        help="测试决策步离线 CE 权重（meta-train；行为仍无反馈）",
    )
    parser.add_argument(
        "--trace-steps",
        action="store_true",
        help="Print per-step debugging details on summary episodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        rngseed=args.seed,
        bs=args.batch_size,
        nbiter=args.nbiter,
        save_every=args.save_every,
        pe=args.print_every,
        nbcues=args.nbcues,
        nb_learn_blocks=args.learn_blocks,
        nb_test_blocks=args.test_blocks,
        supervision_size=args.supervision_size,
        analyze_on_save=args.analyze_on_save,
        learn_shuffle_presentation=not args.no_learn_shuffle,
        baux_learn=args.baux_learn,
        baux_test=args.baux_test,
    )
    np.set_printoptions(precision=5)
    set_seed(config.rngseed)

    output_dir = Path(args.output_dir)
    if args.analyze:
        analysis_dir = (
            Path(args.analysis_dir) if args.analysis_dir else output_dir / "analysis"
        )
        run_analyze_mode(config, args.model_path, analysis_dir)
        return

    train(config, output_dir, trace_steps=args.trace_steps)


if __name__ == "__main__":
    main()
