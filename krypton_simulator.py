import tensorflow as tf
import pandas as pd

from matplotlib import pyplot as plt

from utils.dataloader import dataloader

from simulator.drift import ElectronGenerator
from simulator.light_response import S1Pmt

BATCH_SIZE = 64
WAVEFORM_LENGTH = 48


@profile
def main():

    electron_gen = ElectronGenerator()
    s1_gen       = S1Pmt(n_sensors=12, waveform_length=WAVEFORM_LENGTH)

    loss_func    = tf.keras.losses.MeanSquaredError()
    optimizer    = tf.keras.optimizers.Adam()

    # Load the sipm database:
    sipm_db = pd.read_pickle("new_sipm.pkl")

    dl = dataloader(batch_size=256, max_s1=WAVEFORM_LENGTH)

    for i, batch in enumerate(dl.iterate(1)):
        # Convert to tensorflow:
        batch = { key: tf.convert_to_tensor(batch[key]) for key in batch}

        with tf.GradientTape() as tape:
            generated_s1_image = s1_gen(batch['energy_deposits'])

            # generated_s1_image = tf.squeeze(generated_s1_image)
            # true_total_s1 = tf.reduce_sum(batch['S1Pmt'], axis=1)
            # print(true_total_s1.shape)
            # gen_total_s1 = tf.reduce_sum(generated_s1_image, axis=1)
            # print(gen_total_s1.shape)
            loss = loss_func(batch['S1Pmt'],  generated_s1_image)
            # loss = loss_func(true_total_s1,  generated_s1_image)

        grads = tape.gradient(loss, s1_gen.trainable_variables)

        optimizer.apply_gradients(zip(grads, s1_gen.trainable_variables))
        print(i, loss)

        if i > 1000:
            break

        pass

    # # Loop over the files:
    #     # Loop over events in the file:
    #     for event, this_pmaps in event_reader(pmap_file, kdst_file):


    #         input_tensor = [event["X"], event["Y"], event["Z"], 0.0415575 ]

    #         input_tensor = tf.convert_to_tensor(input_tensor, dtype = tf.float32)
    #         input_tensor = tf.reshape(input_tensor, (1,) + input_tensor.shape)



    #         electrons = electron_gen.generate_electrons(input_tensor)




    #         # This is now the list of primary electrons.
    #         # It needs to be passed into several paths.

    #         # 1: generate S1

    #         target_s1_image = assemble_pmt_image(event, this_pmaps)
    #         target_s1_image = tf.reshape(target_s1_image, (1,) +  target_s1_image.shape)



    #         # 2: Drift the electrons to the EL region, including lifetime and diffusion

    #         # 3: at the EL region, generate the light for each electron cloud.

    #         # 3a: simulate the PMT response

    #         # 4b: simulate the SiPM response

    #         # From here, we merge back together.  Figure out the timing difference, aka S2 -S1.
    #         # in real data, S2 is at a fixed trigger and we back out S1
    #         # in MC, S1 is at a fixed position and we set S2.

    print(batch["S1Pmt"].shape)
    x = range(batch['S1Pmt'].shape[-1])
    plt.plot(x, tf.reduce_sum(batch['S1Pmt'], axis=1)[0], label="True S1")
    plt.plot(x, tf.reduce_sum(generated_s1_image, axis=1)[0], label="Generated")
    plt.legend()
    plt.savefig("1D_S1.pdf")
    plt.show()

    plt.imshow(batch['S1Pmt'][0])
    plt.savefig("TruePMTs.pdf")
    plt.show()


    plt.imshow(generated_s1_image[0])
    plt.savefig("GenPMTs.pdf")
    plt.show()

if __name__ == "__main__":
    main()