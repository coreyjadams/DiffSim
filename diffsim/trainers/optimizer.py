import optax


def build_optimizer(config, params):
    opt_func_name = config.mode.optimizer.name
    optimizer = getattr(optax, opt_func_name)(
        config.mode.learning_rate,
        weight_decay = config.mode.weight_decay
    )

    # optimizer = optax.MultiSteps(optimizer, every_k_schedule=4)

    opt_state = optimizer.init(params)


    return optimizer, opt_state
