from config import MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd

import tensorflow as tf
import numpy

import pathlib

class supervised_trainer:

    def __init__(self, config, monitor_data):

        self.config = config

        # These are just for training:
        self.optimizer   = self.build_optimizer()

        self.loss_func   = self.build_loss_function()

        # Hold one batch of data to make plots while training.
        self.monitor_data = monitor_data


    def comparison_plots(self, simulator, plot_directory):

        from matplotlib import pyplot as plt


        # In this function, we take the monitoring data, run an inference step,
        # And make plots of real vs sim responses.

        # First, run the monitor data through the simulator:
        gen_s2_pmt, gen_s2_si = simulator(self.monitor_data['energy_deposits'])

        # Now, instead of computing loss, we generate plots:


        x_ticks = numpy.arange(550)

        batch_index=0

        pmt_dir = plot_directory / pathlib.Path(f"pmts/")
        pmt_dir.mkdir(parents=True, exist_ok=True)
        for i_pmt in range(12):
            fig = plt.figure(figsize=(16,9))
            plt.plot(x_ticks, gen_s2_pmt[batch_index][i_pmt], label=f"Generated PMT {i_pmt} signal")
            plt.plot(x_ticks, self.monitor_data["S2Pmt"][batch_index][i_pmt], label=f"Real PMT {i_pmt} signal")
            plt.savefig(pmt_dir / pathlib.Path(f"pmt_{i_pmt}.png"))
            plt.legend()
            plt.grid(True)
            plt.close()

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
            gen_s2_pmt, gen_s2_si = simulator(batch['energy_deposits'])
            # print(generated_s2_image.shape)



            # s2_pmt_loss = tf.reduce_sum(tf.pow(batch["S2Pmt"] - gen_s2_pmt, 2.), axis=(1,2))
            # s2_si_loss = tf.reduce_sum(tf.pow(batch["S2Si"] - gen_s2_si, 2.), axis=(1,2,3))
            #
            # s2_pmt_loss = tf.reduce_mean(s2_pmt_loss)
            # s2_si_loss = tf.reduce_mean(s2_si_loss)

            s2_pmt_loss = self.loss_func(batch["S2Pmt"],  gen_s2_pmt)
            s2_si_loss  = self.loss_func(batch["S2Si"],  gen_s2_si)
            loss = s2_pmt_loss
            # loss = s2_si_loss + s2_pmt_loss

        metrics['s2_pmt_loss'] = s2_pmt_loss
        metrics['s2_si_loss'] = s2_si_loss

        # Combine losses

        metrics['loss'] = loss


        if MPI_AVAILABLE:
            tape = hvd.DistributedGradientTape(tape)


        grads = tape.gradient(loss, simulator.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        return metrics
