import sys, os
from concurrent import futures

from abc import ABC, abstractmethod

import numpy

class FileReader(ABC):

    def __init__(self, file_info, shuffle=True):

        self.file_info = file_info

        self.shuffle   = shuffle

        self.length    = None

        self.active_files = []


    @abstractmethod
    def read_event(self, entry_no):
        """
        Read a specific event from this file, assuming it's open
        """

        # In general, this must be thread safe!

        raise NotImplementedError("Must be implemented in child class.")


    def iterate(self):

        order = numpy.arange(self.length)

        if self.shuffle:
            numpy.random.shuffle(order)

        for i in order:
            yield self.read_event(i)

    def __len__(self): return self.length

    def __getitem__(self):
        raise NotImplementedError("Must be implemented in child class.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    @abstractmethod
    def open(self):
        raise NotImplementedError("Must be implemented in child class.")
        return self

    def close(self):
        for f in self.active_files:
            f.close()
        self.active_files = []

from concurrent.futures import ProcessPoolExecutor

import random
class FileLoader(ABC):

    def __init__(self, MPI_AVAILABLE, reader_class, **kwargs):

        self.MPI_AVAILABLE = MPI_AVAILABLE

        # Format should be an iterable of iterables
        if MPI_AVAILABLE:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
        else:
            self.rank = 0


        if self.rank == 0:
            self.file_list = self.discover_files(**kwargs)

        if self.MPI_AVAILABLE:
            self.coordinate(MPI.COMM_WORLD)

        self.epoch = 0

        self.shuffle = True

        self.active  = True

        self.reader_class = reader_class

        # We cache the readers, which store their (sparse) data in memory
        self.readers = []
        # When a reader is finished opening, the next one gets initialized and starts reading

    @abstractmethod
    def discover_files(self, **kwargs):
        raise NotImplementedError("Must be implemented in child class")


    def preload_dataset(self):
        """
        Load the entire dataset by using futures to parallelize it
        """

        futures = []

        with ProcessPoolExecutor(max_workers=32) as executor:
            for files in self.file_list:
                reader = self.reader_class(files, self.shuffle)
                future = executor.submit(
                    reader.open
                )
                futures.append(future)

        self.readers = [ f.result() for f in futures]
        self.lengths = numpy.asarray([ len(r) for r in self.readers ])

        # This makes lookup of indexes easier:
        self.ends    = numpy.cumsum(self.lengths)
        self.starts  = self.ends - self.lengths
        # self.ends    = self.ends

        # And, the total length
        self.length  = numpy.sum(self.lengths)

    def __len__(self): return self.length

    def global_index_to_file_index(self, idx):
        # Figure out the first spot where idx exceeds the start:
        file_index = numpy.argmax(self.ends>idx)

        local_index = idx - self.starts[file_index]

        return file_index, local_index

    def __getitem__(self, idx):

        file_idx, local_idx = self.global_index_to_file_index(idx)

        return self.readers[file_idx][local_idx]


    def iterate(self):

        while self.active:

            # Generically yield the files:
            for files in self.file_list:
                with self.reader_class(files, shuffle=self.shuffle) as this_reader:
                    yield this_reader

            # Increment the epoch
            self.epoch += 1

    def __len__(self): return self.length

    def coordinate(self, COMM):

        # Scater the file list from rank 0

        rank = COMM.Get_rank()
        size = COMM.Get_size()


        if rank == 0:
            # We have to manually reshape it to the right size.
            broadcast_data = []
            step_size = int(len(self.file_list) / size)
            for i in range(size):
                broadcast_data.append(self.file_list[i*step_size : (i+1)*step_size])

            data = broadcast_data
        else:
            data = None

        # print("Initial data: ", data)
        data = COMM.scatter(data, root=0)
        self.file_list = data


class DataPreloader:


    def __init__(self, batch_size, MPI_AVAILABLE, file_loader):

        self.MPI_AVAILABLE = MPI_AVAILABLE
        self.batch_size = batch_size
        # self.max_preload_batches = max_preload_batches

        self.file_loader = file_loader


    def iterate(self):

        output_data_stack = {
            'energy_deposits' : [],
            'S2Si'            : [],
            'S2Pmt'           : [],
        }
        self.active = True
        batch_index = 0

        while self.active:
            # Run this until someone shuts it off

            # The data set is preloaded.  So just iterate over the list:
            for i in range(len(self.file_loader)):

                if not self.active:
                    print("Stopping iterations")
                    raise StopIteration()


                this_output_data = self.file_loader[i]
                if this_output_data is None:
                    continue
                else:
                    for key in this_output_data:
                        output_data_stack[key].append(this_output_data[key])


                # output_data['S2Si'][batch_index][:]
                batch_index += 1

                if batch_index == self.batch_size:
                    output_data = {}
                    output_data['energy_deposits'] =  numpy.stack(output_data_stack['energy_deposits'])
                    output_data['S2Si']  = numpy.zeros(shape = [self.batch_size,] + [47,47,550])
                    output_data['S2Pmt'] = numpy.stack(output_data_stack['S2Pmt'])

                    for i, (x,y,z,val) in enumerate(output_data_stack['S2Si']):
                        output_data['S2Si'][i][x,y,z] = val

                    yield output_data
                    batch_index = 0
                    output_data_stack = {
                        'energy_deposits' : [],
                        'S2Si'            : [],
                        'S2Pmt'           : [],
                    }
