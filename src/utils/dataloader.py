import h5py
import tables
import pathlib
import numpy
import glob

from multiprocessing import Pool



class dataloader:

    def __init__(self, path, run,
        batch_size=32, max_energy_depositions=2, db = None):

        self.batch_size = batch_size
        self.path = path
        self.run  = run

        self.n_pmts         = 12
        self.readout_length = 550
        self.max_energy_depositions = max_energy_depositions

        self.s1_pmt_shape = [self.n_pmts, self.readout_length]
        self.s2_pmt_shape = [self.n_pmts, self.readout_length]
        self.s2_si_shape = [47, 47, self.readout_length]

        if db is None:
            raise Exception("Must provide a DB")
        else:
            self.db_lookup = {
                "x_lookup" : numpy.asarray(db['X']),
                "y_lookup" : numpy.asarray(db['Y']),
                "active"   : numpy.asarray(db['Active']),
            }

        self.active_files = []
        self.active = False

    def shutdown(self):
        self.active = False
        self.close_open_files()

    def iterate(self, epochs : int = 1):
        # Pull together enough inputs to form the next batch.

        # For each batch, we need to assemble:
        # - all the energy depositions - DONE
        # - all the S1 signals         - DONE
        # - all the S2Pmt signals      - DONE
        # - all the S2Sipm signals     - DONE

        self.active = True


        output_data_stack = {
            'energy_deposits' : [],
            'S1Pmt'           : [],
            'S2Si'            : [],
            'S2Pmt'           : [],
        }

        # Loop over the files and pull events until the batch is full
        batch_index = 0

        for pmap_file, kdst_file in self.file_list():

            for event, pmaps in self.event_reader_tables(pmap_file, kdst_file):

                this_output_data = self.build_output_data(
                    event,
                    pmaps,
                    peak_location = 10,
                )

                # Skip if there are any issues
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
                    output_data['S1Pmt'] = numpy.stack(output_data_stack['S1Pmt'])
                    output_data['S2Si']  = numpy.zeros(shape = [self.batch_size,] + self.s2_si_shape)
                    output_data['S2Pmt'] = numpy.stack(output_data_stack['S2Pmt'])

                    for i, (x,y,z,val) in enumerate(output_data_stack['S2Si']):
                        output_data['S2Si'][i][x,y,z] = val



                    yield output_data
                    batch_index = 0
                    output_data_stack = {
                        'energy_deposits' : [],
                        'S1Pmt'           : [],
                        'S2Si'            : [],
                        'S2Pmt'           : [],
                    }


    def build_output_data(self, event, pmaps, peak_location):
        # for event, pmaps in self.event_reader(pmap_file, kdst_file):

        # Create the default empty tensors
        output_data = {
            'energy_deposits' : numpy.zeros(shape = (self.max_energy_depositions, 4), dtype=numpy.float32),
            'S1Pmt'           : numpy.zeros(shape = (self.n_pmts, self.readout_length), dtype=numpy.float32),
            'S2Si'            : (),
            'S2Pmt'           : (),
        }

        output_data['energy_deposits'][0,:] = [event['X'], event['Y'], event['Z'], 0.0415575 ]

        output_data['S1Pmt'][:] = self.assemble_pmt_image(
            event, pmaps, peak_location = 10, max_s1_ticks = self.readout_length)
        output_data['S2Pmt'] = self.assemble_pmt_s2(event, pmaps)

        # Veto empty events:
        if output_data['S2Pmt'] is None: return None

        sipm = self.assemble_sipm_image(event, pmaps, self.db_lookup)
        if sipm is None: return None
        output_data['S2Si'] = sipm
        return output_data

    def file_list(self,
        shuffle       : bool = False):
        '''
        Loop over and yield pairs of connected pmap and kdst files.

        This will loop infinitely, until broken
        '''




        kdst_path = self.path / pathlib.Path(f"{self.run}/kdst/")
        pmap_path = self.path / pathlib.Path(f"{self.run}/pmaps/")

        pmap_format = "pmaps_{:04}_8678_trigger1_v1.2.0_20191122_krbg1600.h5"
        kdst_format = "kdst_{:04}_8678_trigger1_v1.2.0_20191122_krbg.h5"

        # How many total files?
        n_files = len(glob.glob(str(kdst_path / pathlib.Path("*.h5"))))

        if n_files == 0:
            raise Exception("No files found!")

        indexes = list(range(n_files))

        while True:

            if not self.active: break

            if shuffle:
                indexes = numpy.random.shuffle(indexes)

            i = 0
            for file_index in range(n_files):

                kdst = kdst_path / pathlib.Path(kdst_format.format(file_index))
                pmap = pmap_path / pathlib.Path(pmap_format.format(file_index))


                if not kdst.is_file(): continue
                if not pmap.is_file(): continue

                yield pmap, kdst

                if i > 1: break


    def event_reader_tables(self, pmap_file, kdst_file):
        f_trigger1_pmap = tables.File(pmap_file)
        self.active_files.append(f_trigger1_pmap)
        f_trigger1_kdst = tables.File(kdst_file)
        self.active_files.append(f_trigger1_kdst)


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

        # with Pool() as p:
        results = map(s, good_events)


        for e, r in zip(good_events, results):
            if r is not None:
                yield e, r

        # Close open files:
        self.close_open_files()

    def close_open_files(self):

        for f in self.active_files:
            f.close()


    def event_reader(self, pmap_file, kdst_file):
        # Open the files:
        h_trigger1_pmap = h5py.File(pmap_file, 'r')
        h_trigger1_kdst = h5py.File(kdst_file, 'r')

        # Get the events table:
        events = h_trigger1_kdst['DST']['Events']

        # Access S1, S2 tables
        keys = {"S1", "S1Pmt", "S2", "S2Pmt", "S2Si"}
        pmap_tables = {key : h_trigger1_pmap["PMAPS"][key] for key in keys}



        for event in events:
            # Check against S1 and S2 numbers
            if event["nS1"] != 1 or event["nS2"] != 1: continue

            # Check Fiducial:
            if event['Z'] < 20 or event['Z'] > 520: continue

            # Check R:
            if event['X']**2 + event['Y']**2 > 180**2: continue

            event_number = event['event']

            # What does this correspond to in the raw file?
            selection = { key : pmap_tables[key]['event'] == event_number for key in keys }

            this_pmaps = { key : pmap_tables[key][selection[key]] for key in keys}

            yield event, this_pmaps

        h_trigger1_kdst.close()
        h_trigger1_pmap.close()

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
        active_selection   = numpy.take(db_lookup["active"], pmaps["S2Si"]["nsipm"]).astype(numpy.bool)


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


        # print("x_locations: ", x_locations)
        # print("y_locations: ", y_locations)
        # print("z_locations: ", z_locations)
        # print("energy: ", energy)

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

        if (z_locations > self.readout_length).any(): return None

        output_s2pmt = numpy.zeros(shape = self.s2_pmt_shape)

        global_index = 0

        for i_pmt in range(self.n_pmts):
            for i_z in z_locations:
                output_s2pmt[i_pmt, i_z] += s2_values[global_index]

                global_index += 1


        return output_s2pmt


    def assemble_pmt_image(self, event, pmaps, peak_location=10, max_s1_ticks=30):

        pre_padding = peak_location
        post_padding = max_s1_ticks - peak_location


        recorded_waveform_length = len(pmaps['S1'])

        # We create an output array that we'll pack with this data:
        output = numpy.zeros(shape=(self.n_pmts, max_s1_ticks))

        # Whats the peak tick?
        peak_tick = numpy.argmax(pmaps['S1']['ene'])
        # print(peak_tick)

        # First, we figure out where to put the waveform's first peak.
        if peak_tick < pre_padding:
            # the waveform pre-ticks fit entirely, so we just figure out the
            # start point in the output waveform for it.
            output_start = pre_padding - peak_tick
            input_start  = 0
        else:
            # Then, the pre-ticks don't fit entirely.
            output_start = 0
            input_start  = peak_tick - pre_padding

        # Second, figure out where to put the waveform's last peak.
        if recorded_waveform_length - peak_tick < post_padding:
            # Then, everything fits on the back side.  The output goes to the end
            # of the waveform.  So, prepadding + remaining waveform after peak.
            output_end = pre_padding + recorded_waveform_length - peak_tick
            # Input end is however long is left in the waveform after the peak:
            input_end = recorded_waveform_length
        else:
            # The waveform goes too long on the backside.  Truncate.
            # Input end is everything that will fit:
            input_end = peak_tick + post_padding
            # Output end is the entire waveform:
            output_end = output_start + (input_end - input_start)



        # Make sure the first tick used in the waveform puts the peak at pre_padding
        waveform_start = max(peak_tick - pre_padding, 0)
        # Make sure if the wave form is too long we truncate
        # (peak_tick + post_padding < len(waveform) - waveform_start)
        waveform_end   = max(post_padding - peak_tick, post_padding)

        # Loop over the PMTs:
        # offset = int(min_waveform_length/2)
        offset = 0
        for i in range(self.n_pmts):
            indexes = pmaps['S1Pmt']['npmt'] == i
            # print(indexes)
            this_waveform = pmaps['S1Pmt'][indexes]
            # print(this_waveform)

            this_waveform = this_waveform['ene']
            output[i][output_start:output_end] = this_waveform[input_start:input_end]

        return output
