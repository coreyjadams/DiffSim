import tensorflow as tf
import pandas as pd


src_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.dirname(src_dir)
sys.path.insert(0,src_dir)


from matplotlib import pyplot as plt

from utils.dataloader import dataloader

from simulator.NEW_Simulator import NEXT_Simulator




BATCH_SIZE = 64


# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra

@profile
@hydra.main(config_path="config", config_name="config")
def main(cfg : OmegaConf) -> None:

    print(cfg)


    simulator = NEXT_Simulator()   

    loss_func    = tf.keras.losses.MeanSquaredError()
    optimizer    = tf.keras.optimizers.Adam()

    summary      = tf.summary.create_file_writer("log/")

    # Load the sipm database:
    sipm_db = pd.read_pickle("new_sipm.pkl")

    dl = dataloader(batch_size=BATCH_SIZE, db=sipm_db, run=cfg.data.run)

    global_step = tf.Variable(0, dtype=tf.dtypes.int64)

    for i, batch in enumerate(dl.iterate()):
        # Convert to tensorflow:
        batch = { key: tf.convert_to_tensor(batch[key]) for key in batch}


        # # continue
        # print(batch["S2Si"].shape)
        # print(batch["S1Pmt"].shape)

        # Store the true images and labels:
        # store_s1_images(batch, summary, global_step)
        # store_s2_images(batch, summary, global_step)


        # if simulate_s1:

        with tf.GradientTape() as tape:
            generated_s2_image = simulator(batch['energy_deposits'])
            # print(generated_s2_image.shape)

            loss = loss_func(batch["S2Pmt"],  generated_s2_image)
            # loss = loss_func(true_total_s1,  generated_s1_image)

        grads = tape.gradient(loss, simulator.trainable_variables)

        optimizer.apply_gradients(zip(grads, simulator.trainable_variables))

        print(i, loss)

        # if i > 100:
        #     break

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

    # print(batch["S1Pmt"].shape)
    # x = range(batch['S1Pmt'].shape[-1])
    # plt.plot(x, tf.reduce_sum(batch['S1Pmt'], axis=1)[0], label="True S1")
    # plt.plot(x, tf.reduce_sum(generated_s1_image, axis=1)[0], label="Generated")
    # plt.legend()
    # plt.savefig("1D_S1.pdf")
    # plt.show()

    # plt.imshow(batch['S1Pmt'][0])
    # plt.savefig("TruePMTs.pdf")
    # plt.show()


    # plt.imshow(generated_s1_image[0])
    # plt.savefig("GenPMTs.pdf")
    # plt.show()
    # 
    # 


def store_s1_images(batch,summary, step):
    with summary.as_default():
        s1_image = tf.reshape(batch["S1Pmt"], batch["S1Pmt"].shape + (1,))
        tf.summary.image("S1PMT", s1_image, step)

def store_s2_images(batch, summary, step):
    # Store 3 2D projections:
    with summary.as_default():
        # Compress three dimensions:
        for i in range(3):
            # Offset axis by one to skip the first dim
            s2_image = tf.reduce_sum(batch['S2Si'], axis=1+i)
            s2_image = tf.reshape(s2_image, s2_image.shape + (1,))
            tf.summary.image(f"Compression/{i}", s2_image, step)

if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()
