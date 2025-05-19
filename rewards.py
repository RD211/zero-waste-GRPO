def dumb_reward_func(
    completion: str,
    answer: str,
):
    return abs(len(completion) - 512) / 512