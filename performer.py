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
        if note.note >= 0 and note.note < len(self.keys_string):
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
        MIDI_PATH = "data/Everytime_We_Touch.mid"
        piano_roll = midi_filename_to_piano_roll(MIDI_PATH)
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