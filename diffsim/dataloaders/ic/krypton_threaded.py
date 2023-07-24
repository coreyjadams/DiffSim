import os

import pathlib
import tables
import glob

import pandas as pd

import numpy

from . preloader import FileReader, FileLoader


class KryptonReader(FileReader):

    max_energy_depositions = 2
    readout_length = 550
    n_pmts = 12


    def open(self):

        self.s2_pmt_shape = [self.n_pmts, self.readout_length]
        self.s2_si_shape = [47, 47, self.readout_length]


        db = pd.read_pickle("database/new_sipm.pkl")
        self.db_lookup = {
            "x_lookup" : numpy.asarray(db['X']),
            "y_lookup" : numpy.asarray(db['Y']),
            "active"   : numpy.asarray(db['Active']),
        }

        # For Krypton, this preloads the pmaps and kdst and selects the good events:
        events, pmaps = self.event_reader_tables(
            pmap_file = self.file_info["pmap"],
            kdst_file = self.file_info["kdst"],
        )

        self.events = events
        self.pmaps = pmaps

        # print(len(self.events))
        # print(len(self.pmaps))

        self.length = len(self.events)

        return self

    def __getitem__(self, event_no):
        return self.read_event(event_no)

    def read_event(self, event_no):

        assert event_no < self.length

        event = self.events[event_no]
        pmaps = self.pmaps[event_no]

        # Create the default empty tensors
        output_data = {
            'energy_deposits' : numpy.zeros(shape = (self.max_energy_depositions, 4), dtype=numpy.float32),
            'S2Si'            : (),
            'S2Pmt'           : (),
        }

        output_data['energy_deposits'][0,:] = [event['X'], event['Y'], event['Z'], 0.0415575 ]

        output_data['S2Pmt'] = self.assemble_pmt_s2(event, pmaps)

        # Veto empty events:
        if output_data['S2Pmt'] is None: return None

        sipm = self.assemble_sipm_image(event, pmaps, self.db_lookup)
        if sipm is None: return None
        output_data['S2Si'] = sipm
        return output_data




    def assemble_sipm_image(self, event, pmaps, db_lookup):
        # SiPM locations range from -235 to 235 mm in X and Y (inclusive) every 10mm
        # That's 47 locations in X and Y.

        # First, we note the time of S1, which will tell us Z locations
        s1_t = event['S1t'] # This will be in nano seconds

        s2_times = pmaps['S2']['time']
        waveform_length = len(s2_times)

        #This is more sensors than we need, strictly.  Not all of them are filled.

        # For each sensor in the raw waveforms, we need to take the sensor index,
        # look up the X/Y,
        # convert to index, and deposit in the dense data

        # We loop over the waveforms in chunks of (s2_times)


        # Figure out the total number of sensors:
        n_sensors = int(len(pmaps["S2Si"]) / waveform_length)

        # Some events have a bug in the s1 or s2 count.
        # This is an explicit check of that.
        if (pmaps["S2"]["peak"] != 0).any():
            return None


        # Get the energy, and use it to select only active hits
        energy      = pmaps["S2Si"]["ene"]
        energy_selection   = energy != 0.0

        # Make sure we're selecting only active sensors:
        active_selection   = numpy.take(db_lookup["active"], pmaps["S2Si"]["nsipm"]).astype(bool)


        # Merge the selections:
        # selection = numpy.logical_and(energy_selection, active_selection)
        selection = active_selection

        # Each sensor has values, some zero, for every tick in the s2_times.
        # The Z values are constructed from these, so stack this vector up
        # by the total number of unique sensors

        ticks       = numpy.tile(s2_times, n_sensors)[selection]

        # x and y are from the sipm lookup tables, and then filter by active sites
        x_locations = numpy.take(db_lookup["x_lookup"], pmaps["S2Si"]["nsipm"])[selection]
        y_locations = numpy.take(db_lookup["y_lookup"], pmaps["S2Si"]["nsipm"])[selection]

        # Filter the energy to active sites
        energy      = energy[selection]

        # Convert to physical coordinates
        x_locations = ((x_locations / 10 + 23.5) - 1).astype(numpy.int32)
        y_locations = ((y_locations / 10 + 23.5) - 1).astype(numpy.int32)
        z_locations = ((ticks - s1_t) / 1000).astype(numpy.int32)




        return x_locations, y_locations, z_locations, energy


    def assemble_pmt_s2(self, event, pmaps):

        # What is the shape of S1 that we have?
        # print(pmaps['S1'].dtype)
        # print(pmaps['S1'])


        s1_t = event['S1t'] # This will be in nano seconds

        # How to do this?
        # S2Pmt contains a list of non-zero signals, including PMT identifier, with a number of
        # values for each pmt
        # S2 contains the list of time ticks

        # We assume the same time resolution as the sipms:

        s2_times = pmaps['S2']['time']

        # ticks     = numpy.tile(s2_times, self.n_pmts)
        s2_values = pmaps['S2Pmt']['ene']
        pmt_id    = pmaps['S2Pmt']['npmt']

        z_locations = ((s2_times - s1_t) / 1000).astype(numpy.int32)

        if (z_locations > self.readout_length).any() or (z_locations < 0).any(): return None

        output_s2pmt = numpy.zeros(shape = self.s2_pmt_shape)

        global_index = 0

        for i_pmt in range(self.n_pmts):
            for i_z in z_locations:
                output_s2pmt[i_pmt, i_z] += s2_values[global_index]

                global_index += 1


        return output_s2pmt


    def event_reader_tables(self, pmap_file, kdst_file):
        """
        This function reads the entire krypton file into memory upon opening
        """

        f_trigger1_pmap = tables.File(pmap_file)
        # self.active_files.append(f_trigger1_pmap)
        f_trigger1_kdst = tables.File(kdst_file)
        # self.active_files.append(f_trigger1_kdst)


        # Load the whole events table into memory
        events = f_trigger1_kdst.get_node('/DST/Events').read()

        keys = {"S1", "S1Pmt", "S2", "S2Pmt", "S2Si"}
        pmap_tables = {key : f_trigger1_pmap.get_node(f"/PMAPS/{key}/").read() for key in keys}


        def select_good_event_numbers(events_table):

            # Compute the fiducial events:
            good_events_locations = events['nS1'] == 1
            good_events_locations = numpy.logical_and(good_events_locations, events_table['nS2'] == 1)
            good_events_locations = numpy.logical_and(good_events_locations, events_table['Z'] > 20)
            good_events_locations = numpy.logical_and(good_events_locations, events_table['Z'] < 520)
            good_events_locations = numpy.logical_and(good_events_locations, events_table['X']**2 + events_table['Y']**2 < 180.**2)

            return events[good_events_locations]

        good_events = select_good_event_numbers(events)


        def slice_into_event(_pmaps, _event, _keys):
            event_number = _event['event']
            # What does this correspond to in the raw file?
            selection = { key : _pmaps[key]['event'] == event_number for key in _keys }
            this_pmaps = { key : _pmaps[key][selection[key]] for key in keys}

            return this_pmaps

        s = lambda e : slice_into_event(pmap_tables, e, keys)

        good_pmaps = list(map(s, good_events))

        f_trigger1_pmap.close()
        f_trigger1_kdst.close()

        return good_events, good_pmaps




class KryptonLoader(FileLoader):

    def discover_files(self, path, run, trigger=None):
        """
        The role of this function is just to find all the possible files, based on
        path and run.  It doesn't shuffle or coordinate, that is elsewhere
        """

        file_list = []

        kdst_path = pathlib.Path(path) / pathlib.Path(str(run)) / "kdst/"
        pmap_path = pathlib.Path(path) / pathlib.Path(str(run)) / "pmaps/"

        if trigger is not None:
            kdst_path /= f"trigger{trigger}/"
            pmap_path /= f"trigger{trigger}/"

        kdst_prefix  = "kdst_"
        pmap_prefix  = "pmaps_"

        pmap_postfix = "_trigger1_v1.2.0_20191122_krbg1600.h5"
        kdst_postfix = "_trigger1_v1.2.0_20191122_krbg.h5"


        # How many total files?
        glob_str  = str(kdst_path) + "/*.h5"
        kdst_list = glob.glob(glob_str)
        # kdst_list = glob.glob(str(kdst_path / pathlib.Path("*.h5")))
        n_files = len(kdst_list)
        if n_files == 0:
            raise Exception(f"No files found at {glob_str}!")

        i = 0
        for kdst_file in kdst_list:

            # Need to back out the index:
            index_str = kdst_file.replace(str(kdst_path),"")

            index_str = index_str.replace(f"/{kdst_prefix}", "")
            index_str = index_str.replace(f"_{run}{kdst_postfix}", "")

            # if index_str == "1515": continue

            pmap_name = f"{pmap_prefix}{index_str}_{run}{pmap_postfix}"
            pmap_file = pmap_path / pathlib.Path(pmap_name)


            if os.path.isfile(pmap_file):
                file_list.append({
                    "pmap" : str(pmap_file),
                    "kdst" : kdst_file
                })

            i += 1
            if i > 51: break

        return file_list
