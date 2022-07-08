from config import MPI_AVAILABLE

if MPI_AVAILABLE:
    import horovod.tensorflow as hvd

import tensorflow as tf
import numpy

import pathlib
from matplotlib import pyplot as plt



class supervised_trainer:

    def __init__(self, config, monitor_data):

        self.config = config

        # These are just for training:
        self.optimizer   = self.build_optimizer()

        self.loss_func   = self.build_loss_function()

        # Hold one batch of data to make plots while training.
        self.monitor_data = monitor_data



    def plot_pmts(self, plot_dir, sim_pmts, real_pmts):

        x_ticks = numpy.arange(550)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for i_pmt in range(12):
            fig = plt.figure(figsize=(16,9))
            plt.plot(x_ticks, sim_pmts[i_pmt], label=f"Generated PMT {i_pmt} signal")
            plt.plot(x_ticks, real_pmts[i_pmt], label=f"Real PMT {i_pmt} signal")
            plt.legend()
            plt.grid(True)
            plt.xlabel("Time Tick [us]")
            plt.ylabel("Amplitude")
            plt.savefig(plot_dir / pathlib.Path(f"pmt_{i_pmt}.png"))
            plt.tight_layout()
            plt.close()

        return

    # def plot_waveform_nicely(input_waveforms, input_labels):

    #     x_ticks = numpy.arange(input_waveforms[0].shape[0])

    #     # Ratio



    #     fig = plt.figure(figsize=(16,9))
    #     plt.plot(x_ticks, sim_pmts[i_pmt], label=f"Generated PMT {i_pmt} signal")
    #     plt.plot(x_ticks, real_pmts[i_pmt], label=f"Real PMT {i_pmt} signal")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xlabel("Time Tick [us]")
    #     plt.ylabel("Amplitude")
    #     plt.savefig(plot_dir / pathlib.Path(f"pmt_{i_pmt}.png"))
    #     plt.tight_layout()
    #     plt.close()

    def plot_sipms(self, plot_dir, sim_sipms, real_sipms):

        x_ticks = numpy.arange(550)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Find the index of the peak sipm location:
        max_value = tf.reduce_max(real_sipms)
        max_loc   = tf.where(real_sipms == max_value)[0]

        # Draw the response sliced across the three dimensions:
        # Slice on z:    

        # This plots over all z, around the highest value sipm:
        central_i_x = max_loc[0]
        central_i_y = max_loc[1]
        central_i_z = max_loc[2]

        for i_x in [central_i_x -1, central_i_x, central_i_x + 1]:
            if i_x < 0 or i_x >= 47: continue
            for i_y in [central_i_y -1, central_i_y, central_i_y + 1]:
                if i_y < 0 or i_y >= 47: continue

                fig = plt.figure(figsize=(16,9))
                plt.plot(x_ticks, sim_sipms[i_x][i_y], label=f"Generated SiPM [{i_x}, {i_y}] signal")
                plt.plot(x_ticks, real_sipms[i_x][i_y], label=f"Real SiPM [{i_x}, {i_y}] signal")
                plt.legend()
                plt.grid(True)
                plt.xlabel("Time Tick [us]")
                plt.ylabel("Amplitude")
                plt.savefig(plot_dir / pathlib.Path(f"sipm_{i_x}_{i_y}.png"))
                plt.tight_layout()
                plt.close()

        # This plots x and y for a fixed z:
        for i_z in [central_i_z -1, central_i_z, central_i_z + 1]:
            if i_z < 0 or i_z >= 550: continue

            # 
            fig = plt.figure()
            plt.imshow(sim_sipms[:,:,i_z])
            plt.tight_layout()
            plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_slice_{i_z}.png"))
            plt.close()

            fig = plt.figure()
            plt.imshow(real_sipms[:,:,i_z])
            plt.tight_layout()
            plt.savefig(plot_dir / pathlib.Path(f"real_sipm_slice_{i_z}.png"))
            plt.close()


    def plot_compressed_sipms(self, plot_dir, sim_sipms, real_sipms, axis):
        
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # What is the axis label for this compression?
        if axis == 0:
            label = "x"
        elif axis == 1:
            label = "y"
        elif axis == 2:
            label = "z"
        else:
            raise Exception(f"Invalid axis {axis} provided to compression plots.")

        # Compress time ticks:
        sim_comp = tf.reduce_sum(sim_sipms, axis=axis)

        fig = plt.figure()
        plt.imshow(sim_comp)
        plt.tight_layout()
        plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_compress_{label}.png"))
        plt.close()

        # Compress time ticks:
        real_comp = tf.reduce_sum(real_sipms, axis=axis)

        fig = plt.figure()
        plt.imshow(real_comp)
        plt.tight_layout()
        plt.savefig(plot_dir / pathlib.Path(f"real_sipm_compress_{label}.png"))
        plt.close()

    def comparison_plots(self, simulator, plot_directory):



        # In this function, we take the monitoring data, run an inference step,
        # And make plots of real vs sim responses.

        # First, run the monitor data through the simulator:
        gen_s2_pmt, gen_s2_si = simulator(self.monitor_data['energy_deposits'])

        # Save the raw data into a file:
        print(plot_directory)
        plot_directory.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(plot_directory / pathlib.Path(f"output_arrays.npz"),
            real_pmts  = self.monitor_data["S2Pmt"],
            gen_pmts   = gen_s2_pmt,
            real_sipms = self.monitor_data["S2Si"],
            gen_sipms  = gen_s2_si,

            )


        # # Now, instead of computing loss, we generate plots:



        batch_index=0

        pmt_dir = plot_directory / pathlib.Path(f"pmts/")
        self.plot_pmts(pmt_dir, gen_s2_pmt[batch_index], self.monitor_data["S2Pmt"][batch_index])

        sim_data_3d  = gen_s2_si[batch_index]
        real_data_3d = self.monitor_data["S2Si"][batch_index]


        sipm_dir = plot_directory / pathlib.Path(f"sipms/")
        self.plot_sipms(sipm_dir, sim_data_3d, real_data_3d)


        # Take 2D compression views:
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=0)
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=1)
        self.plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=2)





    def build_optimizer(self):
        from config.mode import OptimizerKind

        lr = self.config.learning_rate

        if self.config.optimizer == OptimizerKind.Adam:
            return tf.keras.optimizers.Adam(lr)
        elif self.config.optimizer == OptimizerKind.SGD:
            return tf.keras.optimizers.SGD(lr)
        else:
            raise Exception("Unhandled Optimizer")

    def build_loss_function(self):
        from config.mode import Loss

        if self.config.loss == Loss.MSE:
            return tf.keras.losses.MeanSquaredError()
        else:
            raise Exception("Unhandled Loss Kind")

    def sipm_loss(self, real_sipms, sim_sipms):

        # This loss function is similar to mean absolute error,
        # Except it does not count any location where both input and 
        # output are not zero:
        
        diff = real_sipms - sim_sipms
            
        loss = diff**2

        max_loc = 30
        print("real_sipms: ", real_sipms[0,0,max_loc-5 : max_loc + 5])
        print("sim_sipms: ", sim_sipms[0,0,max_loc-5 : max_loc + 5])
        print("loss: ", loss[0,0,max_loc-5 : max_loc + 5])


        return tf.pow(self.loss_func(real_sipms, sim_sipms), 3)

        non_zero_input = real_sipms != 0
        non_zero_gen   = sim_sipms  != 0

        zero = tf.constant(0, dtype=tf.float32)
        where_input = tf.not_equal(real_sipms, zero)

        joint_locations = tf.math.logical_and(non_zero_input, non_zero_gen)



        # Select the locations from both tensors:
        selected_input = tf.boolean_mask(real_sipms, joint_locations)
        selected_gen   = tf.boolean_mask(sim_sipms, joint_locations)

        return self.loss_func(selected_input, selected_gen)

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

            s2_pmt_loss = self.sipm_loss(batch["S2Pmt"],  gen_s2_pmt)
            # s2_si_loss  = self.sipm_loss(batch["S2Si"],  gen_s2_si)
            loss = s2_pmt_loss
            # loss = s2_si_loss
            # loss = s2_si_loss + s2_pmt_loss

            reg = simulator.regularization()
            # loss += reg

        metrics['loss/s2_pmt_loss'] = s2_pmt_loss
        # metrics['loss/s2_si_loss'] = s2_si_loss
        metrics['loss/regularization'] = reg

        # Combine losses

        metrics['loss/loss'] = loss


        if MPI_AVAILABLE:
            tape = hvd.DistributedGradientTape(tape)


        grads = tape.gradient(loss, simulator.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        return metrics
