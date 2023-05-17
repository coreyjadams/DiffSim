


def model_summary(writer, weights, step):
    return
    #TODO - tree flatten these
    for w, g in zip(weights, gradients):
        writer.add_histogram("weights/"   + w.name, w, global_step=step)

def wavefunction_summary(writer, latest_psi, step):
    writer.add_histogram("psi", latest_psi, global_step=step)


# @tf.function
def summary(writer, metrics, step):
    for key in metrics:
        writer.add_scalar(key, metrics[key], global_step=step)
