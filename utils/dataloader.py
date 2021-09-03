import h5py
import tables
import pathlib
import numpy
import glob

from multiprocessing import Pool

DEFAULT_PATH = pathlib.Path("/Users/corey.adams/data/NEXT/new_raw_data/")
DEFAULT_RUN  = 8678


class dataloader:

    def __init__(self, path=DEFAULT_PATH, run=DEFAULT_RUN, batch_size=32, max_s1 = 30):

        self.batch_size = batch_size
        self.path = path
        self.run  = run

        self.n_pmts     = 12
        self.n_s1_ticks = max_s1

    @profile
    def iterate(self, epochs : int = 1):
        # Pull together enough inputs to form the next batch.

        # For each batch, we need to assemble:
        # - all the energy depositions - TODO
        # - all the S1 signals         - TODO
        # - all the S2Pmt signals      - TODO
        # - all the S2Sipm signals     - TODO

        output_data = {
            'energy_deposits' : numpy.zeros(shape = (self.batch_size, 1, 4)),
            'S1Pmt'           : numpy.zeros(shape = (self.batch_size, self.n_pmts, self.n_s1_ticks))
        }


        # Loop over the files and pull events until the batch is full
        batch_index = 0

        for pmap_file, kdst_file in self.file_list():

            for event, pmaps in self.event_reader_tables(pmap_file, kdst_file):
            # for event, pmaps in self.event_reader(pmap_file, kdst_file):
                output_data['energy_deposits'][batch_index][:] = [event['X'], event['Y'], event['Z'], 0.0415575 ]
                output_data['S1Pmt'][batch_index][:] = self.assemble_pmt_image(
                    event, pmaps, peak_location = 10, max_s1_ticks = self.n_s1_ticks)
                
                batch_index += 1
                if batch_index == self.batch_size:
                    yield output_data
                    batch_index = 0
                    output_data = {
                        'energy_deposits' : numpy.zeros(shape = (self.batch_size, 1, 4)),
                        'S1Pmt'           : numpy.zeros(shape = (self.batch_size, self.n_pmts, self.n_s1_ticks))
                    }



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

        indexes = list(range(n_files))

        while True:
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

    @profile
    def event_reader_tables(self, pmap_file, kdst_file):
        f_trigger1_pmap = tables.File(pmap_file)
        f_trigger1_kdst = tables.File(kdst_file)

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


        f_trigger1_pmap.close()
        f_trigger1_kdst.close()


    @profile
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

    def assemble_sipm_image(self, event, pmaps, sipm_db):
        # SiPM locations range from -235 to 235 mm in X and Y (inclusive) every 10mm
        # That's 47 locations in X and Y.
        
        # First, we note the time of S1, which will tell us Z locations
        s1_t = event['S1t'] # This will be in nano seconds
        
        s2_times = pmaps['S2']['time']
        waveform_length = len(s2_times)

        # How many SiPM ticks are we going to need?  
        n_ticks = 1600
        
        #This is more sensors than we need, strictly.  Not all of them are filled.
        
        # For each sensor in the raw waveforms, we need to take the sensor index, look up the X/Y, 
        # convert to index, and deposit in the dense data
        
        # We loop over the waveforms in chunks of (s2_times)
        
        x, y, z, val = [], [], [], []

        n_sensors = int(len(pmaps["S2Si"]) / waveform_length)
        for i_sensor in range(n_sensors):
            for i_tick, tick in enumerate(s2_times):
                # Global pmaps index repeats every waveform_length
                global_index = i_sensor*waveform_length + i_tick
                # Grab the data for this sensor:
                sensor = pmaps["S2Si"]['nsipm'][global_index]
                ene    = pmaps["S2Si"]['ene'][global_index]
                # Actual time of this data from S2 times list
                time = s2_times[i_tick]
                # What sensor?
                # Verify is active:
                if not sipm_db.iloc[sensor].Active: continue
                
                x.append(sipm_db.iloc[sensor].X)
                y.append(sipm_db.iloc[sensor].Y)
                z.append((time - s1_t) / 1000)
                val.append(ene)

            

        return x, y, z, val


    @profile
    def assemble_pmt_image(self, event, pmaps, peak_location=10, max_s1_ticks=30):

        # What is the shape of S1 that we have?
        # print(pmaps['S1Pmt'].dtype)
        # print(pmaps['S1'].dtype)
        # print(pmaps['S1'])

        pre_padding = peak_location
        post_padding = max_s1_ticks - peak_location

        n_pmts = 12

        recorded_waveform_length = len(pmaps['S1'])

        # We create an output array that we'll pack with this data:
        output = numpy.zeros(shape=(n_pmts, max_s1_ticks))

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


        # print("")
        # print("peak_tick: ", peak_tick)
        # print("recorded_waveform_length: ", recorded_waveform_length)
        # print("pre_padding: ", pre_padding)
        # print("post_padding: ", post_padding)
        # print("max_s1_ticks: ", max_s1_ticks)
        # print("Output from: ", output_start, " to ", output_end)
        # print("input from: ", input_start, " to ", input_end)
        # print("")



        # Make sure the first tick used in the waveform puts the peak at pre_padding
        waveform_start = max(peak_tick - pre_padding, 0)
        # Make sure if the wave form is too long we truncate
        # (peak_tick + post_padding < len(waveform) - waveform_start)
        waveform_end   = max(post_padding - peak_tick, post_padding)

        # Loop over the PMTs:
        # offset = int(min_waveform_length/2)
        offset = 0
        for i in range(n_pmts):
            indexes = pmaps['S1Pmt']['npmt'] == i
            # print(indexes)
            this_waveform = pmaps['S1Pmt'][indexes]
            # print(this_waveform)

            this_waveform = this_waveform['ene']
            output[i][output_start:output_end] = this_waveform[input_start:input_end]

        return output


        # We 

