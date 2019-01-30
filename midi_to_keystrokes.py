#!/usr/bin/env python
#
# test_midiin_callback.py
#
"""Show how to receive MIDI input by setting a callback function."""

from __future__ import print_function

import logging
import sys
import time

from pynput.keyboard import Key, Controller
from rtmidi.midiutil import open_midiinput

log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)


class MidiInputHandler(object):
    def __init__(self, port):
        self.port = port
        self._wallclock = time.time()
        self.keyboard = Controller()
        self.keys_string = '''aksldf;g'h[jq2w3er5t6y7ui]z\\xc,v.b/nm'''
        self.num_keys = len(self.keys_string)
        self.base_note = 12*4 # c4

    def __call__(self, event, data=None):
        message, deltatime = event
        self._wallclock += deltatime
        print("[%s] @%0.6f %r" % (self.port, self._wallclock, message))

        status_byte = message[0]

        if (status_byte >> 4) == 0b1000:
            note_id = message[1] - self.base_note
            print("Note off %d" % note_id)
            if note_id >= 0 and note_id < self.num_keys:
                self.keyboard.release(self.keys_string[note_id])
            
        elif (status_byte >> 4) == 0b1001:
            note_id = message[1] - self.base_note
            print("Note on %d" % note_id)
            if note_id >= 0 and note_id < self.num_keys:
                self.keyboard.press(self.keys_string[note_id])

        # Get first four bits

        # Get first four bits of th

# Prompts user for MIDI input port, unless a valid port number or name
# is given as the first argument on the command line.
# API backend defaults to ALSA on Linux.
port = sys.argv[1] if len(sys.argv) > 1 else None

try:
    midiin, port_name = open_midiinput(port)
except (EOFError, KeyboardInterrupt):
    sys.exit()

print("Attaching MIDI input callback handler.")
midiin.set_callback(MidiInputHandler(port_name))

print("Entering main loop. Press Control-C to exit.")
try:
    # Just wait for keyboard interrupt,
    # everything else is handled via the input callback.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('')
finally:
    print("Exit.")
    midiin.close_port()
    del midiin