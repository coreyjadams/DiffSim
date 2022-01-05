from config import MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd

import tensorflow as tf


class supervised_trainer:

    def __init__(self, config):

        self.config = config

        # These are just for training:
        self.optimizer   = self.build_optimizer()

        self.loss_func   = self.build_loss_function()





    def build_optimizer(self):
        from config.mode import OptimizerKind

        if self.config.optimizer == OptimizerKind.Adam:
            return tf.keras.optimizers.Adam()
        else:
            raise Exception("Unhandled Optimizer")

    def build_loss_function(self):
        from config.mode import Loss

        if self.config.loss == Loss.MSE:
            return tf.keras.losses.MeanSquaredError()
        else:
            raise Exception("Unhandled Loss Kind")

    def train_iteration(self, simulator, batch):

        metrics = {}

        with tf.GradientTape() as tape:
            generated_s2_image = simulator(batch['energy_deposits'])
            # print(generated_s2_image.shape)

            s2_pmt_loss = self.loss_func(batch["S2Pmt"],  generated_s2_image)

        metrics['s2_pmt_loss'] = s2_pmt_loss

        # Combine losses
        loss = s2_pmt_loss

        metrics['loss'] = loss


        if MPI_AVAILABLE:
            tape = hvd.DistributedGradientTape(tape)


        grads = tape.gradient(loss, simulator.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        return metrics
