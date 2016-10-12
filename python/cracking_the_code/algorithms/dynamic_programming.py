
def running_child(steps_left: int, memory: dict = None) -> int:
    if steps_left <= 0:
        return 0
    if steps_left == 1:
        return 1
    if steps_left == 2:
        return 1 + running_child(steps_left - 1)
    if steps_left == 3:
        return 1 + running_child(steps_left - 2) + running_child(steps_left - 1)
    if memory is None:
        memory = {}
    if memory.get(steps_left) is None:
        memory[steps_left] = (running_child(steps_left - 3, memory) +
                              running_child(steps_left - 2, memory) +
                              running_child(steps_left - 1, memory))
    return memory[steps_left]
