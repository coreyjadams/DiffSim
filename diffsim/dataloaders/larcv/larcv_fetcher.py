import os
import time

from . import data_transforms

import numpy


# Functional programming approach to building up the dataset objects:


def generic_meta(zoom_sampling=1.0):

    return numpy.array([
        (
            [
                int(48*zoom_sampling), 
                int(48*zoom_sampling), 
                int(550*zoom_sampling), 
            ], 
            [480., 480., 550.],
            [-235., -235., 0])
        ],
        dtype=[
            ('n_voxels', "int", (3)),
            ('size', "float", (3)),
            ('origin', "float", (3)),
        ]
    )

pmaps_meta  = lambda : generic_meta(zoom_sampling=1.0)
lr_meta     = lambda : generic_meta(zoom_sampling=10.0)
energy_meta = lambda : generic_meta(zoom_sampling=10.0)



def create_larcv_interface(random_access_mode, distributed, seed):

    # Not needed, enforced by data.py
    # if random_access_mode not in ["serial_access", "random_blocks"]:
    #     raise Exception(f"Can not use mode {random_access_mode}")

    if seed == -1:
        seed = int(time.time())
    if distributed:
        from larcv import distributed_queue_interface as queueloader
    else:
        from larcv import queueloader


    larcv_interface = queueloader.queue_interface(
        random_access_mode=str(random_access_mode.name), seed=seed)
    larcv_interface.no_warnings()

    return larcv_interface

def prepare_next_config(batch_size, input_file, data_args, name, is_mc):


    # First, verify the files exist:
    if not os.path.exists(input_file):
        raise Exception(f"File {input_file} not found")


    from larcv.config_builder import ConfigBuilder
    cb = ConfigBuilder()
    cb.set_parameter([str(input_file)], "InputFiles")
    cb.set_parameter(6, "ProcessDriver", "IOManager", "Verbosity")
    cb.set_parameter(6, "ProcessDriver", "Verbosity")
    cb.set_parameter(6, "Verbosity")

    print(data_args)

    # Get the S2Si:
    cb.add_batch_filler(
        datatype  = "sparse3d",
        producer  = data_args.image_key,
        name      = name+"S2Si",
        MaxVoxels = 3000,
        Augment   = False,
        Channels  = [0]
    )

    # Get the PMTs if available:
    cb.add_batch_filler(
        datatype  = "sparse2d",
        producer  = "S2Pmt",
        name      = name+"S2Pmt",
        MaxVoxels = 1000,
        Augment   = False,
        Channels  = [0]
    )

    # Build up the data_keys:
    data_keys = {
        'S2Si': name + 'S2Si',
        'S2Pmt': name + 'S2Pmt',
    }


    if is_mc:

        # Need to convert the clusrer3D energy deps to sparse3d first:
        cb.add_preprocess(
            process = "TensorFromCluster",
            datatype = "cluster3d",
            producer = "mc_hits",
            OutputProducer = "e_deps"
        )
        # Get the energy depositions:
        cb.add_batch_filler(
            datatype  = "sparse3d",
            producer  = "e_deps",
            name      = name+"e_deps",
            MaxVoxels = 300,
            UnfilledVoxelValue = 0.0,
            Augment   = False,
            Channels  = [0]
        )
        data_keys["e_deps"] =name + "e_deps"
    else:
        if data_args.name == "krypton":
            cb.add_batch_filler(
                datatype  = "particle",
                producer  = "event",
                name      = name+"event",
                MaxParticles = 1,
            )
            data_keys["event"] =name + "event"

    # Prepare data managers:
    io_config = {
        'filler_name' : name,
        'filler_cfg'  : cb.get_config(),
        'verbosity'   : 5,
        'make_copy'   : False
    }

    # import json
    # print(json.dumps(cb.get_config(), indent=2))

    return io_config, data_keys



def prepare_interface(batch_size, storage_name, larcv_interface, io_config, data_keys, color=0):

    """
    Not a pure function!  it changes state of the larcv_interface
    """
    larcv_interface.prepare_manager(
        storage_name, io_config, batch_size, data_keys, color=color)
    # This queues up the next data
    # self._larcv_interface.prepare_next(name)

    while larcv_interface.is_reading(storage_name):
        time.sleep(0.01)


    return larcv_interface.size(storage_name)


def create_larcv_dataset(data_args, batch_size, batch_keys,
                         input_file, name,
                         distributed=False, sparse=False):
    """
    Create a new iterable dataset of the file specified in data_args
    pass

    """

    # Create a larcv interface:
    interface = create_larcv_interface(
        random_access_mode = data_args.mode,
        distributed = distributed,
        seed=data_args.seed)


    # Next, prepare the config info for this interface:
    io_config, data_keys =  prepare_next_config(
        batch_size = batch_size,
        data_args  = data_args,
        input_file = input_file,
        name       = name,
        is_mc      = data_args.mc)
    for key in data_keys:
        if key not in batch_keys: batch_keys += [key,]
        
    # Now, fire up the interface:
    prepare_interface(
        batch_size,
        storage_name    = name,
        larcv_interface = interface,
        io_config       = io_config,
        data_keys       = data_keys)


    # Finally, create the iterable object to hold all of this:
    dataset = larcv_dataset(
        larcv_interface = interface,
        batch_keys      = batch_keys,
        name            = name,
        data_args       = data_args,
        is_mc           = data_args.mc,
        sparse          = sparse)


    return dataset

class larcv_dataset(object):
    """ Represents a (possibly distributed) larcv dataset on one file

    Implements __len__ and __iter__ to enable fast, iterable datasets.

    May also in the future implement __getitem__(idx) to enable slower random access.

    """

    def __init__(self, larcv_interface, batch_keys, name, data_args, is_mc=True, sparse=False):
        """
        Init takes a preconfigured larcv queue interface
        """

        self.larcv_interface = larcv_interface
        self.data_args       = data_args
        self.storage_name    = name
        self.batch_keys      = batch_keys
        # self.vertex_depth    = vertex_depth
        # self.event_id        = event_id
        self.sparse          = sparse

        # self.data_keys = data_keys

        # Get image meta:
        self.lr_meta = lr_meta()
        self.pmaps_meta = pmaps_meta()

        self.stop = False

    def __len__(self):
        return self.larcv_interface.size(self.storage_name)

    def __iter__(self):
        return self

    def __next__(self):

        batch = self.fetch_next_batch(self.storage_name, True)
        return batch


    def __del__(self):
        self.stop = True

    def image_size(self, key):
        meta = self.image_meta(key)
        return meta['n_voxels'][0]

    def image_meta(self, key):
        if "S2Si" in key: return pmaps_meta()
        else: return lr_meta()

    def fetch_next_batch(self, name, force_pop=False, event_id=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False


        minibatch_data = self.larcv_interface.fetch_minibatch_data(self.storage_name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self.larcv_interface.fetch_minibatch_dims(self.storage_name)

        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # This brings up the next data to current data
        if pop:
            self.larcv_interface.prepare_next(self.storage_name)


        # Purge unneeded keys:
        minibatch_data = {
            key : minibatch_data[key] for key in minibatch_data if key in self.batch_keys
        }


        for key in minibatch_data.keys():
            if event_id:
                if key == 'entries' or key == 'event_ids':
                    continue

            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        # We need the event id for vertex classification, even if it's not used.
        # if self.event_id or self.vertex_depth is not None:
        if 'label' in minibatch_data.keys():
            label_particle = minibatch_data['label'][:,0]
            minibatch_data['label'] = label_particle['_pdg'].astype("int64")


        if 'energy' in self.batch_keys:
            minibatch_data['energy'] = label_particle['energy_init']


        # Shape the images:

        if not self.sparse:
            for key in self.batch_keys:
                if "S2Si" in key:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_dense_3d(
                        minibatch_data[key],
                        dense_shape = self.pmaps_meta['n_voxels'][0],
                    )
                    print(self.pmaps_meta)
                if "S2Pmt" in key:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_dense_2d(
                        minibatch_data[key],
                        dense_shape = (12,550)
                    )
                if "e_deps" in key:
                    minibatch_data[key] = numpy.squeeze(minibatch_data[key], axis=1)

                    minibatch_data[key] = data_transforms.larcv_edeps(
                        minibatch_data[key], 
                        generic_meta(10.))

                if "event" in key:
                    minibatch_data["e_deps"] = data_transforms.larcv_event_deps(minibatch_data[key])
                    # exit()
                    # Drop the 'event' piece:
                    # print(minibatch_data["e_deps"].shape)
                    minibatch_data.pop(key)

                    # # print(minibatch_data[key].shape)
                    # # print(minibatch_data[key])
                    # print("X: ", numpy.min(minibatch_data[key][:,:,0]), numpy.max(minibatch_data[key][:,:,0]))
                    # print("Y: ", numpy.min(minibatch_data[key][:,:,1]), numpy.max(minibatch_data[key][:,:,1]))
                    # print("Z: ", numpy.min(minibatch_data[key][:,:,2]), numpy.max(minibatch_data[key][:,:,2]))
                    # print("E: ", numpy.min(minibatch_data[key][:,:,3]), numpy.max(minibatch_data[key][:,:,3]))
                    # # minibatch_data[key][:,:,0,] = minibatch_data[key][:,:,0] / 20 - 23.5
                    # # minibatch_data[key][:,:,1,] = minibatch_data[key][:,:,1] / 20 - 23.5
                    # # minibatch_data[key][:,:,3,] = minibatch_data[key][:,:,3] / 20.
                    # # print("X: ", numpy.min(minibatch_data[key][:,:,0]), numpy.max(minibatch_data[key][:,:,0]))
                    # # print("Y: ", numpy.min(minibatch_data[key][:,:,1]), numpy.max(minibatch_data[key][:,:,1]))
                    # # print("Z: ", numpy.min(minibatch_data[key][:,:,2]), numpy.max(minibatch_data[key][:,:,2]))
                    # # print("E: ", numpy.min(minibatch_data[key][:,:,3]), numpy.max(minibatch_data[key][:,:,3]))
                    # exit()
                    
                    # # print(minibatch_data[key])
                    # # exit()

        return minibatch_data
