import optax


def build_optimizer(config, params):
    opt_func_name = config.mode.optimizer.name
    optimizer = getattr(optax, opt_func_name)(
        config.mode.learning_rate,
        weight_decay = config.mode.weight_decay
    )

    opt_state = optimizer.init(params)


    return optimizer, opt_state
