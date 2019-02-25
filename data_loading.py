import numpy as np
import os
import pretty_midi
from scipy.interpolate import interp1d
import torch
import torch.utils.data as data

# Borrowed heavily from
# https://github.com/SudharshanShanmugasundaram/Music-Generation/blob/master/notebooks/musicGeneration.ipynb

def midi_filename_to_piano_roll(midi_filename, beat_divisions=4):

    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    # Use auto-detected tempo to get flattened piano roll
    beat_times = midi_data.get_beats()
    # Interpolate between each entry beat_divisions times evenly
    # Warning: not thoroughly tested yet
    beat_times = np.interp(
        np.linspace(0, len(beat_times), (len(beat_times))*beat_divisions),
        xp=np.linspace(0, len(beat_times), len(beat_times)),
        fp=beat_times)
    # Interpolate down to 16th notes
    piano_roll = midi_data.get_piano_roll(times=beat_times)
    # Pressed notes are replaced by 1
    piano_roll[piano_roll > 0] = 1

    # Tempos:
    tempo_change_times, tempi = midi_data.get_tempo_changes()
    if len(tempo_change_times) > 1:
        tempo_roll = interp1d(tempo_change_times, tempi,
                kind='linear', bounds_error=False,
                fill_value=(tempi[0], tempi[-1])
            )(beat_times)
    else:
        tempo_roll = np.ones(beat_times.shape[0])*tempi[0]

    return piano_roll, tempo_roll


class NotesGenerationDataset(data.Dataset):
    
    def __init__(self, midi_folder_path,
            longest_sequence_length=None,
            lowest_note="C3",
            highest_note="C7"):
        self.lowest_note_num = pretty_midi.note_name_to_number(lowest_note)
        self.highest_note_num = pretty_midi.note_name_to_number(highest_note)
        self.num_notes = self.highest_note_num - self.lowest_note_num
        self.midi_folder_path = midi_folder_path
        # TODO(gizatt) Make this recursive within the training
        # folder for better organization.
        midi_filenames = os.listdir(midi_folder_path)
        self.longest_sequence_length = longest_sequence_length
        midi_full_filenames = map(
            lambda filename: os.path.join(
                midi_folder_path, filename),midi_filenames)
        self.midi_full_filenames = list(midi_full_filenames)
        if longest_sequence_length is None:
            self.update_the_max_length()

        # Preload all data. We'll never have *that* much.
        def load_piano_roll(filename):
            piano_roll, tempos = midi_filename_to_piano_roll(
                filename)
            # Cut down to the selected note range
            return piano_roll[self.lowest_note_num:self.highest_note_num, :]
        self.piano_rolls = [load_piano_roll(filename) for filename in self.midi_full_filenames]

    def update_the_max_length(self):
        sequences_lengths = map(lambda filename: 
            midi_filename_to_piano_roll(filename)[0].shape[1],
            self.midi_full_filenames)
        max_length = max(sequences_lengths)
        self.longest_sequence_length = max_length

    def __len__(self):
        return len(self.midi_full_filenames)
    
    def __getitem__(self, index):
        piano_roll = self.piano_rolls[index]
        
        # Shifting by one time step
        sequence_length = piano_roll.shape[1] - 1
        
        # Shifting by one time step
        input_sequence = piano_roll[:, :-1]
        ground_truth_sequence = piano_roll[:, 1:]
        # padding sequence so that all of them have the same length,
        # padding with zeros.
        # TODO(gizatt): What if I use mode="wrap" here? I'll have less
        # "wasted" training data space and not encourage doing absolutely
        # nothing (which will be an attractor for a network trained with
        # long empty stretches?), but will have weird edge effects after
        # cadences / finishing phrases?
        input_sequence_padded = np.pad(
            input_sequence,
            pad_width=((0, 0), 
              (0, self.longest_sequence_length-input_sequence.shape[1])),
            mode="constant",
            constant_values=0).T

        ground_truth_sequence_padded = np.pad(
            ground_truth_sequence,
            pad_width=((0, 0), 
              (0, self.longest_sequence_length-input_sequence.shape[1])),
            mode="constant",
            constant_values=-100).T
        
        return (torch.FloatTensor(input_sequence_padded),
                torch.LongTensor(ground_truth_sequence_padded),
                torch.LongTensor([sequence_length]) )

    
def post_process_sequence_batch(batch_tuple):
    # Reorder sample batches into
    # order of descending sequence length,
    # as required by an eventual call to
    # pack_padded_sequence in the RNN.

    input_sequences, output_sequences, lengths = batch_tuple
    
    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)
    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)
    
    input_sequence_batch_sorted = input_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    
    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)
    
    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)
    
    return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)
