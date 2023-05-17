import optax


def build_optimizer(config, params):
    optimizer = optax.adam(config.mode.learning_rate)
    opt_state = optimizer.init(params)


    return optimizer, opt_state