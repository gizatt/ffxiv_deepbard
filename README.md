## TODO / Notes

Note generator model:
- RNN from https://github.com/SudharshanShanmugasundaram/Music-Generation is
implemented but doesn't produce good results yet.  I didn't copy over
their learning rate tuning, and haven't gotten good sanity
checks on the training procedure yet, so maybe I should try training it
against trivial examples (e.g. very repetitive examples that I synthesize
myself) to sanity-check it.
    - Why can't I load from saved state-dict?
    - Copy over their learning rate search stuff, and understand the subtleties
of their implementation (like how the masking works, why gradient
clipping, other stuff from http://warmspringwinds.github.io/pytorch/rnns/2018/01/27/learning-to-generate-lyrics-and-music-with-recurrent-neural-networks/).
    - Can I somehow constrain their model to only produce single voice lines?
- Use wavenet style dilated convolutions for note input.
- Add a "memory" module to store state using something like an LSTM.

Playback system:
- Piano roll refactor is halfway done, but seems to have some bugs.
- Given past N piano slices, predicting which note in the next slice
is melody is a pretty reasonable supervised learning problem -- generate
data by data augmentation, primarily, by taking truly monophonic midis
and adding relatively random noise.