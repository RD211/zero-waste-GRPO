def dumb_reward_func(
    completion: str,
    answer: str,
):
    return -len(completion) / 4096