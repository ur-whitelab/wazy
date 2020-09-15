import numpy as np
import tensorflow as tf
import random

# expect to be pointed to a single big .npy file with all the
# one-hot encoded peptides in it
def load_data(filename, labels_filename=None):
    with open(filename, 'rb') as f:
        all_peps = np.load(f)
    
    return all_peps


def prepare_data(positive_filename, negative_filenames, weights = None):
    # set-up weights
    # not used yet
    # load randomly shuffled training data
    
    positive_peps = load_data(positive_filename)
    if type(negative_filenames) != list:
        negative_filenames = [negative_filenames]
    negative_peps = []
    for n in negative_filenames:
        negative_peps.extend(load_data(n))
    # now downsample without replacement
    if len(negative_peps) < len(positive_peps):
        print('Unable to find enough negative examples for {}'.format(positive_filename))
        print('Using', *[n for n in negative_filenames])
        exit(1)
    negative_peps = random.sample(negative_peps, k=len(positive_peps))
    # convert negative peps into array
    # I still don't get numpy...
    # why do I have to create a newaxis here? Isn't there an easier way?
    negative_peps = np.concatenate([n[np.newaxis, :, :] for n in negative_peps], axis=0)
    peps = np.concatenate([positive_peps, negative_peps])
   
    # calculate activities if we're not doing regression
    labels = np.zeros(len(peps)) # one-hot encoding of labels
    labels[:len(positive_peps)] += 1. # positive labels are 1

    return tf.data.Dataset.from_tensor_slices((peps, labels))
