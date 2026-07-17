import numpy as np


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


def stronger_cue_index(i: int, j: int) -> int:
    """真实全序：编号越小越强（0 最强）。"""
    return min(i, j)


def weaker_cue_index(i: int, j: int) -> int:
    return max(i, j)


def default_true_rank(nbcues: int) -> list[int]:
    """从强到弱的 cue 编号列表（与 stronger_cue_index 一致）。"""
    return list(range(nbcues))


def sample_random_true_rank(nbcues: int, rng=None) -> list[int]:
    """返回一个随机排列，表示 cue 索引从强到弱的顺序。

    例如 [3,0,5,1,2,6,4,7] 表示 cue 3 最强，cue 7 最弱。
    该排列将抽象「位置」（0 最强，nbcues-1 最弱）映射为实际 cue 编号。
    """
    if rng is None:
        rng = np.random
    return [int(x) for x in rng.permutation(nbcues)]
