from collections import namedtuple
import IPython
import numpy as np
import pretty_midi
import pynput
from pynput.keyboard import Key, Controller
from threading import Thread, Lock
import time

from data_loading import midi_filename_to_piano_roll

default_keys_string = '''aksldf;g'h[jq2w3er5t6y7ui]z\\xc,v.b/nm'''

class MonophonicPerformer():
    ''' Allows thread-safe queueing of future piano roll
    slices (quantized in terms of notes and tempo), and manages
    real-time playing of the queued slices against the system clock.
    Manages priotiziation of voices to play. '''
    # Base note of C4
    def __init__(self, keys_string=default_keys_string, base_note=12*4,
                 beat_divisions=4, buffer_size=1024, default_tempo=120):
        self.keys_string = keys_string
        self.beat_divisions = beat_divisions
        self.num_keys = len(self.keys_string)
        self.playing = False
        self.buffer_size = buffer_size
        self.note_buffer = np.zeros(self.num_keys, self.buffer_size)
        self.tempo_buffer = np.ones(self.buffer_size)*default_tempo
        self.lock = Lock()
        self.playing_thread = None
        self.note_buffer_head = 0
        self.keyboard = Controller()

    def set_piano_roll_slice(self, slice_index, piano_slice, tempo=None):
        if piano_slice.shape != (self.num_keys,):
            raise ValueError("Slice was wrong shape: got %s" % piano_slice.shape)
        with self.lock:
            if slice_index <= self.note_buffer_head:
                print("This slice is in the past, skipping.")
                return
            write_ind = (self.note_buffer_head + slice_index) % self.num_keys
            self.note_buffer[:, write_ind] = piano_roll[:]
            if tempo is not None:
                self.tempo_buffer[write_ind] = tempo

    def decideAndExecuteAction(self, current_slice, last_slice, last_note):
        # Dumb baseline:
        # If the last note was nothing or is not continued,
        # take the highest note in the current slice.
        if last_note is None or current_slice[last_note] == 0:
            candidates = np.nonzero(current_slice)
            if len(candidates) > 0:
                new_note = candidates[-1]
            else:
                new_note = None
        else: 
            # Otherwise continue the current note
            new_note = last_note
        
        # Release events
        if last_note is not None and new_note is not last_note:
            self.keyboard.release(self.keys_string[last_note])
        
        # Press events
        if new_note is not None and new_note is not last_note:
            self.keyboard.press(self.keys_string[new_note])

        return new_note

    def start_playing(self):
        self.playing = True
        self.playing_thread = Thread(target=self.play)
        self.playing_thread.start()

    def play(self):
        current_slice = np.zeros(self.num_keys)
        last_note_time = time.time()
        last_note = None
        while self.playing:
            with self.lock:
                last_slice = current_slice
                current_slice = self.note_buffer[:, self.note_buffer_head]
                current_tempo = self.tempo_buffer[self.note_buffer_head]
                self.note_buffer_head = (self.note_buffer_head + 1) % self.num_keys

            # Given current and last note to play, decide current action.
            last_note = self.decideAndExecuteAction(current_slice, last_slice, last_note)
            # Tempo is in BPM
            next_note_time = last_note_time + 60. / (current_tempo * self.beat_divisions)
            last_note_time = time.time()
            t = next_note_time - time.time()
            if t > 0:
                time.sleep(t)

    def wait_for_stop_playing(self, final_index):
        while True:
            with self.lock:
                if self.note_buffer_head >= final_index:
                    break
        self.stop_playing()

    def stop_playing(self):
        self.playing = False
        if self.playing_thread:
            self.playing_thread.join()

if __name__ == "__main__":
    try:
        MIDI_PATH = "data/Everytime_We_Touch.mid"
        piano_roll, tempo_roll = midi_filename_to_piano_roll(MIDI_PATH)
        midi_data = pretty_midi.PrettyMIDI(MIDI_PATH)
        
        notes = []
        for instrument_num in [0, 1]:
            print("Instrument %d has %d notes" % (instrument_num,len(midi_data.instruments[instrument_num].notes)))
            notes += midi_data.instruments[instrument_num].notes
        print("%d notes total" % len(notes))

        # Clear out notes that are lower simultaneous with other notes
        good_notes_mask = []
        for note_i in range(len(notes)):
            this_start = notes[note_i].start
            this_end = notes[note_i].end
            this_pitch = notes[note_i].pitch
            reject = False
            for other_note_i in range(len(notes)):
                if note_i == other_note_i:
                    continue
                if (notes[other_note_i].start >= this_start - 0.01 and
                    notes[other_note_i].start <= this_start + 0.2 and
                    notes[other_note_i].pitch > this_pitch):
                    reject = True
                    print("Note %s, rejecting bc note %s" % (str(notes[other_note_i]), str(notes[note_i])))
                    continue
            if not reject:
                good_notes_mask.append(note_i)

        notes = [notes[x] for x in good_notes_mask]
        notes = sorted(notes, key=lambda x: x.start)
        intervals = np.array([[note.start, note.end] for note in notes])
        pitches = np.array([note.pitch for note in notes])

        c4 = 12*4
        performer = MonophonicPerformer(base_note=c4)
        for i in range(len(notes)):
            performer.enqueue(
                NoteQueueEntry(
                    note=pitches[i]-12*4,
                    start=intervals[i][0],
                    stop=intervals[i][1]))

        print("STARTING...")
        performer.start_playing()
        performer.wait_for_stop_playing()

    except KeyboardInterrupt as e: 
        print("Keyboard interrupt, shutting down.")
    performer.stop_playing()