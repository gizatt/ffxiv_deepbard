## TODO / Notes

- Piano roll refactor is halfway done, but seems to have some bugs.
- Given past N piano slices, predicting which note in the next slice
is melody is a pretty reasonable supervised learning problem -- generate
data by data augmentation, primarily, by taking truly monophonic midis
and adding relatively random noise.