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

NoteQueueEntry = namedtuple('NoteQueueEntry', ['start', 'stop', 'note'])

class MonophonicPerformer():
    # Base note of C4
    def __init__(self, keys_string=default_keys_string, base_note=12*4):
        self.keys_string = keys_string
        self.playing = False
        # Each queue entry should be an interval (start, stop)
        # and a note (range 0, len(default_keys_string))
        self.note_queue = []
        self.lock = Lock()
        self.playing_thread = None
        self.keyboard = Controller()

    def enqueue(self, note):
        assert(isinstance(note, NoteQueueEntry))
        assert(note.note >= 0 and note.note < len(self.keys_string))
        with self.lock:
            self.note_queue.append(note)

    def start_playing(self):
        self.start_time = time.time()
        self.playing = True
        self.playing_thread = Thread(target=self.play)
        self.playing_thread.start()

    def play(self):
        while self.playing:
            noteQueueEntry = None
            with self.lock:
                if len(self.note_queue) > 0:
                    noteQueueEntry = self.note_queue.pop(0)
            # Play note, if we have one.
            if noteQueueEntry is not None:
                t = time.time() - self.start_time
                if noteQueueEntry.start - t > 0:
                    time.sleep(noteQueueEntry.start - t)
                self.keyboard.press(self.keys_string[noteQueueEntry.note])
                t = time.time() - self.start_time
                if noteQueueEntry.stop - t > 0:
                    time.sleep(noteQueueEntry.stop - t)
                self.keyboard.release(self.keys_string[noteQueueEntry.note])
            time.sleep(0.01)

    def wait_for_stop_playing(self):
        while True:
            with self.lock:
                if len(self.note_queue) == 0:
                    break
        self.stop_playing()

    def stop_playing(self):
        self.playing = False
        if self.playing_thread:
            self.playing_thread.join()

if __name__ == "__main__":
    try:
        MIDI_PATH = "data/FF6Boss.mid"
        piano_roll = midi_filename_to_piano_roll(MIDI_PATH)
        midi_data = pretty_midi.PrettyMIDI(MIDI_PATH)
        instrument_num = 0
        notes = midi_data.instruments[instrument_num].notes
        intervals = np.array([[note.start, note.end] for note in notes])
        pitches = np.array([note.pitch for note in notes])

        c4 = 12*4
        performer = MonophonicPerformer(base_note=c4)
        performer.start_playing()
        for i in range(len(notes)):
            performer.enqueue(
                NoteQueueEntry(
                    note=pitches[i]-12*4,
                    start=intervals[i][0],
                    stop=intervals[i][1]))
            time.sleep(0.1)
        performer.wait_for_stop_playing()

    except KeyboardInterrupt as e: 
        print("Keyboard interrupt, shutting down.")
    performer.stop_playing()