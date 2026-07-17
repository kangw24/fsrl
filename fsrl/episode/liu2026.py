"""Liu 2026 符号化 support/query episode 运行逻辑。

关键约束:
  - Support phase: 学习固定 8 pairs，仅展示，无决策
  - Query phase: 测试全部 28 pairs，无反馈
  - 外循环损失来自 query phase 的交叉熵

图片身份抽象为 cue 符号；有符号差值表示当次柱条公开呈现的关系信息。
该 trial-local observation 是合法的 distance-aware 条件，但完整 true_rank、
query 答案和人类选择不得进入模型。sign-only 条件必须单独比较。
"""


import numpy as np
import torch
import torch.nn.functional as F

from fsrl.device import DEVICE, log
from fsrl.episode.types import EpisodeRecord, EpisodeStats, TestResponse
from fsrl.task.constants import NUMRESPONSESTEP
from fsrl.task.cues import sample_random_true_rank
from fsrl.task.liu2026 import (
    build_liu2026_query_set,
    build_liu2026_step_inputs,
    build_liu2026_support_set,
    generate_liu2026_cue_data,
    map_published_support_pair_to_cues,
    prepare_liu2026_query_trial,
    prepare_liu2026_support_trial,
)


def _position_pairs_to_canonical_cue_pairs(position_pairs, true_rank):
    """Map published weak-to-strong position pairs to episode cue pairs."""
    return {
        (
            min(cue_i, cue_j),
            max(cue_i, cue_j),
        )
        for cue_i, cue_j, _ in (
            map_published_support_pair_to_cues(pair, true_rank)
            for pair in position_pairs
        )
    }


def _is_retro_latent_rank_model(net):
    """判断 net 是否为 RetroLatentRank 混合模型。"""
    return hasattr(net, "rnn") and hasattr(net, "latent_rank") and hasattr(
        net, "encode_support_sequence"
    )


def _is_rnn_model(net):
    """判断 net 是否为 RetroModulRNN/VanillaRNN 风格的 RNN 模型。"""
    return (
        hasattr(net, "initialZeroState")
        and hasattr(net, "initialZeroPlasticWeights")
        and not hasattr(net, "encode_support_trial")
        and not _is_retro_latent_rank_model(net)
    )


def _is_latent_rank_model(net):
    """判断 net 是否为 LatentRankMetaLearner。"""
    return (
        hasattr(net, "encode_support_trial")
        and hasattr(net, "infer_latent_rank")
        and not _is_retro_latent_rank_model(net)
    )


def _is_q_learning_model(net):
    """判断 net 是否为 QLearningBaseline。"""
    return hasattr(net, "q_values") and hasattr(net, "choose")


def _blank_cue_vec(config):
    """blank step 使用的零 cue 向量。"""
    return np.zeros(config.cs * 2, dtype="float32")


def _masked_observation_sign(teacher_sign, trial_mask):
    """Return a scalar sign when everyone observed it, otherwise a per-subject sign.

    A zero entry means that the support outcome was not available to that batch
    element.  Process heads must treat zero as a missing edge, not as a tie.
    """
    if trial_mask is None or bool(torch.all(trial_mask > 0.0)):
        return teacher_sign
    return trial_mask * float(teacher_sign)


def _aggregate_latent_rank_context(
    config, net, cue_data, support_trials, pair_to_idx=None, drop_mask=None,
    training=False, true_rank=None, rng=None
):
    """对 LatentRankMetaLearner，重放 support trials 并推断 rank scores。"""
    trial_encs = []
    support_observations = []
    for trial_info in support_trials:
        cue_i, cue_j, teacher_sign = trial_info
        cue_vecs = []
        for batch_index in range(config.bs):
            cv, _, _ = prepare_liu2026_support_trial(
                config, cue_data, trial_info, batch_index, rng=rng
            )
            cue_vecs.append(cv)
        cue_vecs_t = torch.tensor(np.array(cue_vecs), dtype=torch.float32, device=DEVICE)
        enc = net.encode_support_trial(cue_vecs_t)
        if drop_mask is not None and pair_to_idx is not None:
            pair_idx = pair_to_idx[(min(cue_i, cue_j), max(cue_i, cue_j))]
            enc = enc * drop_mask[:, pair_idx][:, None]
        trial_encs.append(enc)
        trial_mask = None
        if drop_mask is not None and pair_to_idx is not None:
            pair_idx = pair_to_idx[(min(cue_i, cue_j), max(cue_i, cue_j))]
            trial_mask = drop_mask[:, pair_idx]
        support_observations.append(
            (cue_i, cue_j, _masked_observation_sign(teacher_sign, trial_mask))
        )

    trial_encs = torch.stack(trial_encs)  # [num_support, bs, context_dim]
    context = net.aggregate_support_set(trial_encs)
    subject_idx = torch.arange(config.bs, device=DEVICE)
    true_rank_t = None
    if true_rank is not None and getattr(config, "allow_ground_truth_rank_input", False):
        true_rank_t = torch.tensor(true_rank, dtype=torch.long, device=DEVICE)

    # Phase 4 / 过程模型：eval 或 sequential rank update / pairwise accumulator 时把 support evidence 传下去
    obs_for_infer = None
    trial_enc_for_infer = None
    if getattr(net, "use_bayesian_posterior_eval", False) and not training:
        obs_for_infer = support_observations
    if getattr(config, "use_sequential_rank_update", False):
        obs_for_infer = support_observations
        trial_enc_for_infer = trial_encs
    if getattr(config, "use_pairwise_preference_accumulator", False):
        obs_for_infer = support_observations
        trial_enc_for_infer = trial_encs

    rank_scores = net.infer_latent_rank(
        context,
        subject_idx=subject_idx,
        training=training,
        true_rank=true_rank_t,
        support_observations=obs_for_infer,
        trial_encodings=trial_enc_for_infer,
    )
    return context, rank_scores


def _build_support_drop_mask(config, rng, support_cue_pairs):
    """为每个被试生成 supervision drop mask。

    Args:
        support_cue_pairs: list of (cue_i, cue_j)，当前 episode 的实际 cue 对。

    Returns:
        pair_to_idx: dict, canonical cue pair -> index in support_cue_pairs
        drop_mask: [bs, n_support_pairs] tensor，0 表示该 pair 被 drop
    """
    pair_to_idx = {}
    for idx, (i, j) in enumerate(support_cue_pairs):
        pair_to_idx[(min(i, j), max(i, j))] = idx

    n_pairs = len(support_cue_pairs)
    drop_mask = torch.ones(config.bs, n_pairs, device=DEVICE)
    supervision_drop = getattr(config, "supervision_drop", 0)
    if supervision_drop > 0:
        for b in range(config.bs):
            drop_idx = rng.choice(
                n_pairs, size=min(supervision_drop, n_pairs), replace=False
            )
            drop_mask[b, drop_idx] = 0.0
    return pair_to_idx, drop_mask


def _run_support_phase(config, net, cue_data, support_trials, rng=None, training=False, support_cue_pairs=None, true_rank=None):
    """执行 support phase，返回 RNN 状态（如适用）。"""
    if rng is None:
        rng = np.random

    if support_cue_pairs is None:
        # 兼容旧调用：从 trial 中推导（顺序可能不固定，但映射本身有效）
        seen = set()
        support_cue_pairs = []
        for i, j, _ in support_trials:
            canonical = (min(i, j), max(i, j))
            if canonical not in seen:
                seen.add(canonical)
                support_cue_pairs.append(canonical)

    if true_rank is None:
        # 兼容旧行为：cue 编号即位置
        cue_to_pos = {cue: cue for cue in range(config.nbcues)}
    else:
        cue_to_pos = {int(cue): pos for pos, cue in enumerate(true_rank)}

    pair_to_idx, drop_mask = _build_support_drop_mask(config, rng, support_cue_pairs)
    subject_support_pairs = {
        batch_index: [
            (int(pair[0]), int(pair[1]))
            for pair_index, pair in enumerate(support_cue_pairs)
            if float(drop_mask[batch_index, pair_index].item()) > 0.0
        ]
        for batch_index in range(config.bs)
    }
    support_metadata = {"subject_support_pairs": subject_support_pairs}

    if _is_rnn_model(net):
        hidden = net.initialZeroState(config.bs)
        et = net.initialZeroET(config.bs)
        pw = net.initialZeroPlasticWeights(config.bs)

        for trial_info in support_trials:
            cue_i, cue_j, teacher_sign = trial_info
            cue_vecs = []
            teacher_das = []
            deltas = []
            for batch_index in range(config.bs):
                cv, td, _ = prepare_liu2026_support_trial(
                    config, cue_data, trial_info, batch_index, rng=rng
                )
                cue_vecs.append(cv)
                teacher_das.append(td)
                # 当次柱条呈现的符号化相对差值：+ 表示左强，- 表示右强。
                # 使用 cue 在当前 ranking 中的位置差，而非 cue 编号差。
                rank_diff = abs(cue_to_pos[cue_i] - cue_to_pos[cue_j])
                deltas.append(td * rank_diff / max(config.nbcues - 1, 1))

            cue_vecs = np.array(cue_vecs, dtype="float32")
            teacher_da = torch.tensor(teacher_das, dtype=torch.float32, device=DEVICE).view(
                config.bs, 1
            )
            delta_vec = np.array(deltas, dtype="float32")

            # 应用 per-subject supervision drop
            pair_idx = pair_to_idx[(min(cue_i, cue_j), max(cue_i, cue_j))]
            trial_mask = drop_mask[:, pair_idx]  # [bs]
            teacher_da = teacher_da * trial_mask[:, None]
            delta_vec = delta_vec * trial_mask.cpu().numpy()

            for numstep in range(config.support_triallen):
                numstep_ep = 0  # support phase 时间进度可简化为 0
                cue_vec = cue_vecs if numstep == 0 else np.zeros_like(cue_vecs)
                inputs = torch.tensor(cue_vec, dtype=torch.float32, device=DEVICE)
                # 将原始 cue 向量扩展为完整输入向量，携带秩差
                inputs = _expand_inputs(
                    config, inputs, numstep, numstep_ep, delta=delta_vec
                )
                _, _, _, hidden, et, pw = net(inputs, hidden, et, pw, teacher_da=teacher_da)

            # 按 trial 截断计算图，防止 support phase 图过大导致 OOM
            hidden = hidden.detach()
            et = et.detach()
            pw = pw.detach()

        return {
            "hidden": hidden,
            "et": et,
            "pw": pw,
            **support_metadata,
        }

    if _is_retro_latent_rank_model(net):
        # RetroLatentRank：用 RNN 编码整个 support sequence，再 infer rank scores
        inputs_seq = []
        teacher_da_seq = []
        support_observations = []

        for trial_info in support_trials:
            cue_i, cue_j, teacher_sign = trial_info
            cue_vecs = []
            teacher_das = []
            deltas = []
            for batch_index in range(config.bs):
                cv, td, _ = prepare_liu2026_support_trial(
                    config, cue_data, trial_info, batch_index, rng=rng
                )
                cue_vecs.append(cv)
                teacher_das.append(td)
                rank_diff = abs(cue_to_pos[cue_i] - cue_to_pos[cue_j])
                deltas.append(td * rank_diff / max(config.nbcues - 1, 1))

            cue_vecs = np.array(cue_vecs, dtype="float32")
            teacher_da = torch.tensor(
                teacher_das, dtype=torch.float32, device=DEVICE
            ).view(config.bs, 1)
            delta_vec = np.array(deltas, dtype="float32")

            # 应用 per-subject supervision drop
            pair_idx = pair_to_idx[(min(cue_i, cue_j), max(cue_i, cue_j))]
            trial_mask = drop_mask[:, pair_idx]
            teacher_da = teacher_da * trial_mask[:, None]
            delta_vec = delta_vec * trial_mask.cpu().numpy()
            support_observations.append(
                (
                    int(cue_i),
                    int(cue_j),
                    _masked_observation_sign(teacher_sign, trial_mask),
                )
            )

            for numstep in range(config.support_triallen):
                numstep_ep = 0
                cue_vec = cue_vecs if numstep == 0 else np.zeros_like(cue_vecs)
                inputs = torch.tensor(cue_vec, dtype=torch.float32, device=DEVICE)
                inputs = _expand_inputs(
                    config, inputs, numstep, numstep_ep, delta=delta_vec
                )
                inputs_seq.append(inputs)
                # 只在 trial 的第一步提供 teacher_da，后续 blank step 不提供
                teacher_da_seq.append(teacher_da if numstep == 0 else None)

        return_hidden_seq = (
            getattr(config, "use_sequential_rank_update", False)
            or getattr(config, "use_pairwise_preference_accumulator", False)
        )
        encode_out = net.encode_support_sequence(
            inputs_seq, teacher_da_seq, training=training, return_hidden_seq=return_hidden_seq
        )
        if return_hidden_seq:
            context, hidden_seq = encode_out
        else:
            context = encode_out
            hidden_seq = None

        subject_idx = torch.arange(config.bs, device=DEVICE)
        true_rank_t = None
        if true_rank is not None and getattr(config, "allow_ground_truth_rank_input", False):
            true_rank_t = torch.tensor(true_rank, dtype=torch.long, device=DEVICE)

        obs_for_infer = None
        if (
            (getattr(net, "use_bayesian_posterior_eval", False) and not training)
            or getattr(config, "use_sequential_rank_update", False)
            or getattr(config, "use_pairwise_preference_accumulator", False)
        ):
            obs_for_infer = support_observations

        rank_scores = net.infer_latent_rank(
            context,
            subject_idx=subject_idx,
            training=training,
            true_rank=true_rank_t,
            support_observations=obs_for_infer,
            trial_encodings=hidden_seq,
        )
        return (context, rank_scores, support_metadata)

    if _is_latent_rank_model(net):
        # LatentRank 在 support phase 不更新 RNN 状态，直接聚合 context
        context, rank_scores = _aggregate_latent_rank_context(
            config, net, cue_data, support_trials, pair_to_idx, drop_mask,
            training=training, true_rank=true_rank, rng=rng
        )
        return (context, rank_scores, support_metadata)

    if _is_q_learning_model(net):
        # Q-learning 每 episode 重置 Q values
        net.reset(config.bs)
        for trial_info in support_trials:
            cue_i, cue_j, teacher_sign = trial_info
            # 缩放后的秩差：D(i,j) = teacher_sign * |pos_i-pos_j| / (nbcues-1)
            # 对齐论文 D(m,n) = (m-n)/7，这里 teacher_sign=+1 表示 i 强于 j
            rank_diff = abs(cue_to_pos[cue_i] - cue_to_pos[cue_j])
            delta_rank = rank_diff / max(config.nbcues - 1, 1)
            observed_diff = torch.full(
                (config.bs,), teacher_sign * delta_rank, dtype=torch.float32, device=DEVICE
            )
            pair_idx = pair_to_idx[(min(cue_i, cue_j), max(cue_i, cue_j))]
            observed_diff = observed_diff * drop_mask[:, pair_idx]
            net.update_from_pair(cue_i, cue_j, observed_diff)
        return support_metadata

    raise TypeError(f"Unsupported model type: {type(net)}")


def _expand_inputs(
    config,
    cue_batch,
    numstep,
    numstep_ep,
    previous_action=None,
    is_go=False,
    delta=0.0,
):
    """将 [bs, 2*cs] 的 cue batch 扩展为 [bs, inputsize]。"""
    bs = cue_batch.shape[0]
    inputs = torch.zeros(bs, config.inputsize, device=DEVICE)
    inputs[:, : config.cs * 2] = cue_batch
    if is_go:
        inputs[:, config.cs * 2] = 1.0
    inputs[:, config.cs * 2 + 1] = 1.0  # bias
    inputs[:, config.cs * 2 + 2] = numstep_ep / max(config.eplen, 1)
    if isinstance(delta, np.ndarray):
        delta = torch.from_numpy(delta).to(DEVICE)
    inputs[:, config.cs * 2 + 4] = delta  # 当次公开关系的 distance-aware 表示
    if previous_action is not None and numstep == NUMRESPONSESTEP + 1:
        for b in range(bs):
            inputs[b, config.cs * 2 + 5 + previous_action[b]] = 1.0
    return inputs


def _run_query_phase(config, net, cue_data, support_state, training=False, rng=None, query_pairs=None, support_cue_pairs=None):
    """执行 query phase，返回 loss、准确率统计和响应记录。

    training=True 时：
      - RNN 模型按 trial 立即 backward（图太深，不能整 episode 保留）。
      - LatentRank 累积 loss 后一次性 backward（support 图共享）。
      - Q-learning 不可导，不 backward。
    """
    if rng is None:
        rng = np.random

    if query_pairs is None:
        query_pairs = build_liu2026_query_set(config.nbcues)
    query_responses = []
    total_loss_value = 0.0
    total_loss_tensor = torch.tensor(0.0, device=DEVICE)
    num_query_trials = 0

    correct_total = 0
    correct_learned = 0
    correct_nonlearned = 0
    count_learned = 0
    count_nonlearned = 0

    if support_cue_pairs is None:
        support_cue_pairs = list(config.support_pairs)
    learned_pairs = set(support_cue_pairs) | set((j, i) for i, j in support_cue_pairs)

    # 初始化模型相关状态
    if _is_rnn_model(net):
        hidden = support_state["hidden"]
        et = support_state["et"]
        pw = support_state["pw"]
    elif _is_latent_rank_model(net) or _is_retro_latent_rank_model(net):
        context = support_state[0]
        rank_scores = support_state[1]
        hidden = None
    elif _is_q_learning_model(net):
        pass
    else:
        raise TypeError(f"Unsupported model type: {type(net)}")

    # LatentRank / Hybrid 辅助损失（diversity loss 等），不依赖具体 query trial
    aux_loss = torch.tensor(0.0, device=DEVICE)
    if training and (_is_latent_rank_model(net) or _is_retro_latent_rank_model(net)):
        aux_loss = net.get_aux_loss()
        if aux_loss.item() != 0.0:
            total_loss_tensor = total_loss_tensor + aux_loss

    # query 阶段决策温度异质性（per-subject，per-episode）
    choice_temp = getattr(config, "choice_temperature_heterogeneity", 0.0)
    temperature = None
    if choice_temp > 0.0:
        T_np = np.exp(rng.normal(0.0, choice_temp, size=config.bs))
        temperature = torch.from_numpy(T_np).float().to(DEVICE)

    numstep_ep = config.num_support_trials * config.support_triallen

    for block_id in range(config.query_blocks):
        block_pairs = list(query_pairs)
        rng.shuffle(block_pairs)

        for pair in block_pairs:
            strong_cue, weak_cue = pair

            # 每个 trial 统一决定是否交换左右（跨 batch 一致），
            # 既满足左右随机化，又避免 latent_rank 接口的 per-sample cue_pair 限制。
            swap_left_right = rng.rand() < 0.5

            # 准备 cue 向量
            cue_vecs = []
            correct_choices = []
            model_pairs = []
            canonical_pairs = []
            left_is_firsts = []
            for batch_index in range(config.bs):
                cv, correct_choice, model_pair, canonical_pair, left_is_first = (
                    prepare_liu2026_query_trial(
                        config, cue_data, pair, batch_index, swap_left_right=swap_left_right
                    )
                )
                cue_vecs.append(cv)
                correct_choices.append(correct_choice)
                model_pairs.append(model_pair)
                canonical_pairs.append(canonical_pair)
                left_is_firsts.append(left_is_first)
            cue_vecs = torch.tensor(np.array(cue_vecs, dtype="float32"), device=DEVICE)
            correct_choice = np.array(correct_choices, dtype=np.int64)
            model_pair_batch = model_pairs[0]

            previous_action = None
            activout = None
            q_choices = None
            decision_logits = None
            decision_probs = None

            for numstep in range(config.query_triallen):
                if (
                    getattr(config, "analytic_pairwise_control", False)
                    and numstep != NUMRESPONSESTEP
                ):
                    # The analytic control has no query-time recurrent state.
                    # Avoid replaying the same pairwise equation on blank steps.
                    numstep_ep += 1
                    continue
                is_go = numstep == NUMRESPONSESTEP
                cue_batch = cue_vecs if numstep == 0 else torch.zeros_like(cue_vecs)
                inputs = _expand_inputs(
                    config, cue_batch, numstep, numstep_ep, previous_action, is_go
                )

                if _is_rnn_model(net):
                    activout, _, _, hidden, et, pw = net(inputs, hidden, et, pw)
                elif _is_latent_rank_model(net) or _is_retro_latent_rank_model(net):
                    net_kwargs = {
                        "context": context,
                        "rank_scores": rank_scores,
                        "cue_pair": model_pair_batch,
                        "phase": "query",
                        "support_cue_pairs": support_cue_pairs,
                    }
                    if getattr(net, "use_probabilistic_ranking_eval", False):
                        net_kwargs["block_id"] = block_id
                    activout, hidden, _, rank_scores = net(inputs, hidden, **net_kwargs)
                elif _is_q_learning_model(net):
                    activout = torch.zeros(config.bs, 2, device=DEVICE)
                    if numstep == NUMRESPONSESTEP:
                        activout = net.choice_logits(
                            model_pair_batch[0], model_pair_batch[1], temperature=temperature
                        )

                if numstep == NUMRESPONSESTEP:
                    if activout is not None and temperature is not None:
                        activout = activout / temperature[:, None]
                    if _is_q_learning_model(net):
                        decision_logits = activout
                        decision_probs = F.softmax(decision_logits, dim=-1)
                        q_choices = torch.multinomial(decision_probs, 1).squeeze(-1)
                        previous_action = q_choices.detach().cpu().numpy()
                    else:
                        decision_logits = activout
                        decision_probs = F.softmax(decision_logits, dim=-1)
                        choices = torch.multinomial(decision_probs, 1).squeeze(-1)
                        previous_action = choices.detach().cpu().numpy()

                numstep_ep += 1

            # 记录结果
            correct = (previous_action == correct_choice).astype(np.float32)
            correct_total += correct.sum()

            is_learned = pair in learned_pairs or (pair[1], pair[0]) in learned_pairs
            if is_learned:
                correct_learned += correct.sum()
                count_learned += config.bs
            else:
                correct_nonlearned += correct.sum()
                count_nonlearned += config.bs

            # 把 action（0=选左，1=选右）转回 canonical_pair（i<j）上的 prefers_i_over_j
            # 同时记录选择置信度/RT 代理（当 probs 可用时）
            if decision_probs is not None:
                p_chosen = decision_probs.gather(1, torch.from_numpy(previous_action).long().to(DEVICE).unsqueeze(1)).squeeze(1)
                confidences = (2.0 * (p_chosen - 0.5).abs()).detach().cpu().numpy()
                rt_proxies = (-np.log(confidences + 1e-10)).astype(np.float64)
            else:
                confidences = None
                rt_proxies = None

            for batch_index in range(config.bs):
                action = int(previous_action[batch_index])
                canonical_pair = canonical_pairs[batch_index]
                left_is_first = left_is_firsts[batch_index]
                # action=0 表示选左边 cue；左边 cue 是 canonical_pair[0] 当且仅当 left_is_first
                prefers_i_over_j = (action == 0) == left_is_first
                query_responses.append(
                    TestResponse(
                        block_id=block_id,
                        pair=canonical_pair,
                        action=action,
                        prefers_i_over_j=prefers_i_over_j,
                        correct=bool(correct[batch_index]),
                        batch_index=batch_index,
                        confidence=float(confidences[batch_index]) if confidences is not None else None,
                        rt_proxy=float(rt_proxies[batch_index]) if rt_proxies is not None else None,
                    )
                )

            # query CE loss（外循环目标）
            if (
                config.baux_test > 0
                and decision_logits is not None
                and decision_logits.requires_grad
            ):
                targets = torch.from_numpy(correct_choice).long().to(DEVICE)
                trial_loss = config.baux_test * F.cross_entropy(
                    decision_logits, targets
                )
                if training:
                    if _is_rnn_model(net):
                        # RNN：按 trial backward 并截断图
                        trial_loss.backward()
                    elif (
                        _is_latent_rank_model(net)
                        or _is_retro_latent_rank_model(net)
                        or _is_q_learning_model(net)
                    ):
                        # LatentRank / RetroLatentRank：累积 support/query 共享图，最后一次性 backward
                        total_loss_tensor = total_loss_tensor + trial_loss
                total_loss_value += float(trial_loss.detach().cpu())

            num_query_trials += 1

            # 防止 OOM
            if _is_rnn_model(net):
                pw = pw.detach()
                hidden = hidden.detach()
                et = et.detach()

    if training and (
        _is_latent_rank_model(net)
        or _is_retro_latent_rank_model(net)
        or _is_q_learning_model(net)
    ) and total_loss_tensor.item() != 0.0:
        total_loss_tensor.backward()

    avg_loss = torch.tensor(total_loss_value / max(num_query_trials, 1), device=DEVICE)
    objective_value = total_loss_value + float(aux_loss.detach().cpu())

    test_perf = correct_total / max(num_query_trials * config.bs, 1)
    test_perf_learned = correct_learned / max(count_learned, 1)
    test_perf_nonlearned = correct_nonlearned / max(count_nonlearned, 1)

    return {
        "loss": avg_loss,
        "loss_objective": objective_value,
        "test_perf": float(test_perf),
        "test_perf_learned": float(test_perf_learned),
        "test_perf_nonlearned": float(test_perf_nonlearned),
        "query_responses": query_responses,
    }


def run_liu2026_episode(config, net, rng=None, print_trace=False, support_order=None):
    """运行一个完整的 Liu 2026 episode（训练用）。

    Returns:
        EpisodeStats: 包含 loss、query accuracy 等
    """
    cue_data = generate_liu2026_cue_data(config, rng)
    if getattr(config, "randomize_true_rank", False):
        true_rank = sample_random_true_rank(config.nbcues, rng)
    else:
        true_rank = np.arange(config.nbcues, dtype=int)
    support_trials = build_liu2026_support_set(
        config, true_rank, support_order=support_order, rng=rng
    )
    query_pairs = build_liu2026_query_set(config.nbcues, true_rank)
    support_cue_pairs = sorted(
        {(min(i, j), max(i, j)) for i, j, _ in support_trials}
    )

    if print_trace:
        log(f"[liu2026 episode] true_rank={true_rank}, support trials: {len(support_trials)}")

    support_state = _run_support_phase(
        config, net, cue_data, support_trials, rng=rng, training=True,
        support_cue_pairs=support_cue_pairs, true_rank=true_rank,
    )
    query_result = _run_query_phase(
        config, net, cue_data, support_state, training=True, rng=rng,
        query_pairs=query_pairs, support_cue_pairs=support_cue_pairs,
    )

    final_pw = support_state.get("pw") if isinstance(support_state, dict) else None
    if final_pw is None:
        final_pw = torch.zeros(1, device=DEVICE)

    return EpisodeStats(
        loss=query_result["loss"],
        loss_value=float(query_result["loss"].detach().cpu()),
        loss_objective=float(query_result["loss_objective"]),
        test_reward_mean=query_result["test_perf"],
        nbtesttrials=config.num_query_trials,
        test_perf=query_result["test_perf"],
        test_perf_adjacent=None,
        test_perf_nonadjacent=None,
        final_pw=final_pw,
        final_subject_rank=None,
        test_perf_learned=query_result["test_perf_learned"],
        test_perf_nonlearned=query_result["test_perf_nonlearned"],
    )


def _extract_subject_modes(net, bs: int):
    """从 LatentRank/RetroLatentRank 中提取每个被试最终选用的 mode index。"""
    lr = None
    if hasattr(net, "latent_rank"):
        lr = net.latent_rank
    elif hasattr(net, "infer_latent_rank"):
        lr = net
    if lr is None or not hasattr(lr, "_last_selected_mode"):
        return {}
    selected = lr._last_selected_mode
    if selected is None:
        return {}
    selected = selected.detach().cpu().numpy()
    return {i: int(selected[i]) for i in range(bs)}


def run_liu2026_eval_episode(config, net, rng=None, support_order=None):
    """运行一个完整的 Liu 2026 episode（评估用），返回 EpisodeRecord 供分析。"""
    cue_data = generate_liu2026_cue_data(config, rng)
    if getattr(config, "randomize_true_rank", False):
        true_rank = sample_random_true_rank(config.nbcues, rng)
    else:
        true_rank = np.arange(config.nbcues, dtype=int)
    support_trials = build_liu2026_support_set(
        config, true_rank, support_order=support_order, rng=rng
    )
    query_pairs = build_liu2026_query_set(config.nbcues, true_rank)
    # support_cue_pairs 从实际使用的 support trials 推导，支持删除/重排 pair 的诊断
    support_cue_pairs = sorted(
        {(min(i, j), max(i, j)) for i, j, _ in support_trials}
    )

    support_state = _run_support_phase(
        config, net, cue_data, support_trials, rng=rng, training=False,
        support_cue_pairs=support_cue_pairs, true_rank=true_rank,
    )
    query_result = _run_query_phase(
        config, net, cue_data, support_state, training=False, rng=rng,
        query_pairs=query_pairs, support_cue_pairs=support_cue_pairs,
    )

    subject_modes = _extract_subject_modes(net, config.bs)
    if isinstance(support_state, dict):
        subject_support_pairs = support_state.get("subject_support_pairs", {})
    else:
        subject_support_pairs = support_state[2].get("subject_support_pairs", {})

    # 过程模型改造：记录实际使用的 support order 与被删除的 pairs
    # config.support_pairs are ordinal-position pairs, whereas support_cue_pairs
    # are cue-ID pairs.  Compare them only after mapping the configured pairs
    # through this episode's randomized true rank.
    all_config_cue_pairs = _position_pairs_to_canonical_cue_pairs(
        config.support_pairs, true_rank
    )
    dropped_pairs = sorted(all_config_cue_pairs - set(support_cue_pairs))

    return EpisodeRecord(
        nbcues=config.nbcues,
        supervision_set=support_cue_pairs,
        query_set=query_pairs,
        test_responses=query_result["query_responses"],
        true_rank=true_rank,
        subject_true_ranks={i: list(true_rank) for i in range(config.bs)},
        subject_modes=subject_modes,
        support_order=[(int(i), int(j), int(s)) for i, j, s in support_trials],
        support_dropped_pairs=dropped_pairs,
        subject_support_pairs=subject_support_pairs,
        test_perf=query_result["test_perf"],
        test_perf_adjacent=None,
        test_perf_nonadjacent=None,
    )
