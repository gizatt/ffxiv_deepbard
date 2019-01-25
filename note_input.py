import pynput
from pynput.keyboard import Key, Controller
import time

keyboard = Controller()

# Starting at C_N, going up to C_N+2
keys = '''aksldf;g'h[jq2w3er5t6y7ui]z\\xc,v.b/nm'''

print("Start...")
time.sleep(2.0)
print("GO!")

def play_note(key, on, off, verbose=True):
    if verbose:
        print("%s:%f:%f" % (key, on, off))
    keyboard.press(key)
    time.sleep(on)
    keyboard.release(key)
    time.sleep(off)

for on_time in [0.1, 0.05, 0.025]:
    for k in [0, 5, 9, 7]:
        steps = [2, 2, 3, 5]
        step_i = 0
        while k < len(keys):
            play_note(keys[k], on_time, 0.05)
            k += steps[step_i]
            step_i = (step_i + 1) % len(steps)

        for _ in range(2):
            step_i = (step_i - 1) % len(steps)
            k -= steps[step_i]
        while k >= 0:
            play_note(keys[k], on_time, 0.05)
            step_i = (step_i - 1) % len(steps)
            k -= steps[step_i]
## Straight run up keys
# for key in keys:
#     print(key)
#     keyboard.press(key)
#     time.sleep(0.15)
#     keyboard.release(key)
#     time.sleep(0.0)
print("DONE.")