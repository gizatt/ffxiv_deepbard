from collections import namedtuple
import IPython
import matplotlib.pyplot as plt
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
    def __init__(self, keys_string=default_keys_string,
                 beat_divisions=4, buffer_size=1000, default_tempo=90):
        self.keys_string = keys_string
        self.beat_divisions = beat_divisions
        self.num_keys = len(self.keys_string)
        self.playing = False
        self.buffer_size = buffer_size
        self.note_buffer = np.zeros((self.num_keys, self.buffer_size))
        self.tempo_buffer = np.ones(self.buffer_size)*default_tempo
        self.lock = Lock()
        self.playing_thread = None
        self.note_buffer_head = 0
        self.keyboard = Controller()

    def set_piano_roll_slice(self, slice_index, piano_slice, tempo=None):
        if piano_slice.shape != (self.num_keys,):
            raise ValueError("Slice was wrong shape: got %s" % piano_slice.shape)
        with self.lock:
            if slice_index < self.note_buffer_head:
                print("This slice is in the past, skipping.")
                return
            elif (slice_index >= self.note_buffer_head + self.buffer_size):
                print("This slice is too far in the future, skipping.")
            write_ind = slice_index % self.buffer_size
            self.note_buffer[:, write_ind] = piano_slice[:]
            if tempo is not None:
                self.tempo_buffer[write_ind] = tempo

    def decideAndExecuteAction(self, current_slice, last_slice, last_note):
        # Dumb baseline:
        # Take the highest note in the current slice.
        # print("Slice: ", current_slice)
        if not np.any(current_slice):
            new_note = None
        else:
            candidates = np.nonzero(current_slice)
            new_note = candidates[-1][0]    
        
        # print("Last note: %s, curr note: %s" % (str(last_note), str(new_note)))
        # Release events
        if last_note != None and new_note != last_note:
            #print("Releasing")
            self.keyboard.release(self.keys_string[last_note])
        
        # Press events
        if new_note != None and new_note != last_note:
            #print("Pressing")
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
                read_index = self.note_buffer_head % self.buffer_size
                current_slice = self.note_buffer[:, read_index].copy()
                self.note_buffer[:, read_index]*=0
                current_tempo = self.tempo_buffer[read_index]
                self.note_buffer_head += 1

            # Tempo is in BPM
            next_note_time = last_note_time + 60. / (current_tempo * self.beat_divisions)
            t = next_note_time - time.time()
            while t > 0:
                t = next_note_time - time.time()

            # Given current and last note to play, decide current action.
            last_note_time = time.time()
            last_note = self.decideAndExecuteAction(current_slice, last_slice, last_note)

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
        MIDI_PATH = "data/FF1Matoya.mid"
        piano_roll, tempo_roll = midi_filename_to_piano_roll(MIDI_PATH)
        midi_data = pretty_midi.PrettyMIDI(MIDI_PATH)

        print("Piano roll shape: ", piano_roll.shape)
        print("Tempo roll shape: ", tempo_roll.shape)

        c4 = 12*4
        c7 = 12*7

        #plt.figure().set_size_inches(12, 12)
        #plt.imshow(piano_roll[c4:(c7+1), 0:200])
        #plt.show()
        
        performer = MonophonicPerformer()
        for i in range(piano_roll.shape[1]):
            performer.set_piano_roll_slice(i, piano_roll[c4:(c7+1), i], tempo_roll[i])
        print("STARTING...")
        performer.start_playing()
        performer.wait_for_stop_playing(i+1)

    except KeyboardInterrupt as e: 
        print("Keyboard interrupt, shutting down.")
    performer.stop_playing()