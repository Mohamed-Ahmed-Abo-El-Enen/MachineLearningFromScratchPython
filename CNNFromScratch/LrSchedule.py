def lr_schedule(learning_rate, iteration):
    if iteration == 0:
        return learning_rate
    if (iteration >= 0) and (iteration <= 10000):
        return learning_rate
    if iteration > 10000:
        return learning_rate * 0.1
    if iteration > 30000:
        return learning_rate * 0.1