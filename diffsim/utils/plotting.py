import matplotlib
from matplotlib import pyplot as plt

import pathlib

import jax.numpy as numpy

def comparison_plots(plot_directory, simulated_data, real_data):

    # In this function, we take the monitoring data, run an inference step,
    # And make plots of real vs sim responses.


    # Save the raw data into a file:
    plot_directory.mkdir(parents=True, exist_ok=True)

    # Now, we generate plots:


    batch_index=0

    pmt_dir = plot_directory / pathlib.Path(f"pmts/")
    plot_pmts(pmt_dir, simulated_data["S2Pmt"][batch_index], real_data["S2Pmt"][batch_index])

    sim_data_3d  = simulated_data["S2Si"][batch_index]
    real_data_3d = real_data["S2Si"][batch_index]


    sipm_dir = plot_directory / pathlib.Path(f"sipms/")
    plot_sipms(sipm_dir, sim_data_3d, real_data_3d)


    # Take 2D compression views:
    plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=0)
    plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=1)
    plot_compressed_sipms(sipm_dir, sim_data_3d, real_data_3d, axis=2)





def plot_pmts(plot_dir, sim_pmts, real_pmts):

    # x_ticks = numpy.arange(550)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for i_pmt in range(12):

        # Find the peak of this PMT and only plot the nearby data:
        peak_tick = real_pmts[i_pmt].argmax()

        print(f"PMT {i_pmt}")
        print(f" - Real peak: {real_pmts[i_pmt][peak_tick]:.3f} at {peak_tick}")
        print(f" - Sim value: {sim_pmts[i_pmt][peak_tick]:.3f} at {peak_tick}")
        sim_peak = sim_pmts[i_pmt].argmax()
        print(f" - Sim peak: {sim_pmts[i_pmt][sim_peak]:.3f} at {sim_peak}")

        start = max(peak_tick - 50, 0)
        end = min(peak_tick + 50, 550)
        print("start: ", start)
        print("end: ", end)

        x_ticks = numpy.arange(start, end)

        fig = plt.figure(figsize=(16,9))
        plt.plot(x_ticks, sim_pmts[i_pmt][start:end], label=f"Generated PMT {i_pmt} signal")
        plt.plot(x_ticks, real_pmts[i_pmt][start:end], label=f"Real PMT {i_pmt} signal")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Time Tick [us]")
        plt.ylabel("Amplitude")
        plt.savefig(plot_dir / pathlib.Path(f"pmt_{i_pmt}.png"))
        plt.tight_layout()
        plt.close()

    return


def plot_sipms(plot_dir, sim_sipms, real_sipms):

    # x_ticks = numpy.arange(550)
    plot_dir.mkdir(parents=True, exist_ok=True)


    # Find the index of the peak sipm location:
    max_value = numpy.max(real_sipms)
    max_x, max_y, max_z = numpy.unravel_index(numpy.argmax(real_sipms), real_sipms.shape)

    # This plots over all z, around the highest value sipm:
    for i_x in [max_x -1, max_x, max_x + 1]:
        if i_x < 0 or i_x >= 47: continue
        for i_y in [max_y -1, max_y, max_y + 1]:
            if i_y < 0 or i_y >= 47: continue

            # Select the up-to-100 nearest points for plotting:
            start = max(max_z - 50, 0)
            end = min(max_z + 50, 550)

            x_ticks = numpy.arange(start, end)


            # print(sim_sipms[i_x][i_y][max_z-5:max_z+5])
            # print(real_sipms[i_x][i_y][max_z-5:max_z+5])

            fig = plt.figure(figsize=(16,9))
            plt.plot(x_ticks, sim_sipms[i_x][i_y][start:end], label=f"Generated SiPM [{i_x}, {i_y}] signal")
            plt.plot(x_ticks, real_sipms[i_x][i_y][start:end], label=f"Real SiPM [{i_x}, {i_y}] signal")
            plt.legend()
            plt.grid(True)
            plt.xlabel("Time Tick [us]")
            plt.ylabel("Amplitude")
            plt.savefig(plot_dir / pathlib.Path(f"sipm_{i_x}_{i_y}.png"))
            plt.tight_layout()
            plt.close()

    # This plots x and y for a fixed z:
    for i_z in [max_z -1, max_z, max_z + 1]:
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


def plot_compressed_sipms(plot_dir, sim_sipms, real_sipms, axis):

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
    sim_comp = numpy.sum(sim_sipms, axis=axis)

    fig = plt.figure()
    plt.imshow(sim_comp)
    plt.tight_layout()
    plt.savefig(plot_dir / pathlib.Path(f"sim_sipm_compress_{label}.png"))
    plt.close()

    # Compress time ticks:
    real_comp = numpy.sum(real_sipms, axis=axis)

    fig = plt.figure()
    plt.imshow(real_comp)
    plt.tight_layout()
    plt.savefig(plot_dir / pathlib.Path(f"real_sipm_compress_{label}.png"))
    plt.close()
