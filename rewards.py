def dumb_reward_func(
    completion: str,
    answer: str,
):
    return -len(completion)/512.0

def length_reward_func(
    completion: str,
    answer: str,
):
    return min(len(completion)/1024.0, 10.0)

def thinking_reward_func(
    completion: str,
    answer: str,
):
    # Check if the completion contains the <think> tag
    if completion.count("<think>") > 0 and completion.count("</think>") > 0:
        return 1.0
    else:
        return -1.0
    
def accuracy_reward_func(
    completion: str,
    answer: str,
):
    # We extract the answer from boxed{answer}.
    if "oxed{" in completion and "}" in completion:
        start = completion.index("oxed{") + len("oxed{")
        end = completion[start:].index("}") + start
        answer_completion = completion[start:end]
        # Check if the answer is correct
        if answer_completion.strip() == str(answer).strip():
            print("Answer is correct")
            return 1.0
        else:
            return -1.0
    return -1.0