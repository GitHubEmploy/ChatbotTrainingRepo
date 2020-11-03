import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import time, datetime
import os
import tensorflow_datasets as tfds
import pickle
import sys

import os
import io
import pickle
import time
import bz2
import numpy as np

#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
#tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('GPU'))

class TextLoader():
    # Call this class to load text from a file.
    def __init__(self, data_dir, batch_size, seq_length):
        # TextLoader remembers its initialization arguments.
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tensor_sizes = []

        self.tensor_file_template = os.path.join(data_dir, "data{}.npz")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        sizes_file = os.path.join(data_dir, "sizes.pkl")

        self.input_files = self._get_input_file_list(data_dir)
        self.input_file_count = len(self.input_files)

        if self.input_file_count < 1:
            raise ValueError("Input files not found. File names must end in '.txt' or '.bz2'.")

        if self._preprocess_required(vocab_file, sizes_file, self.tensor_file_template, self.input_file_count):
            # If either the vocab file or the tensor file doesn't already exist, create them.
            t0 = time.time()
            print("Preprocessing the following files:")
            for i, filename in enumerate(self.input_files): print("   {}.\t{}".format(i+1, filename))
            print("Saving vocab file")
            self._save_vocab(vocab_file)

            for i, filename in enumerate(self.input_files):
                t1 = time.time()
                print("Preprocessing file {}/{} ({})... ".format(i+1, len(self.input_files), filename),
                        end='', flush=True)
                self._preprocess(self.input_files[i], self.tensor_file_template.format(i))
                self.tensor_sizes.append(self.tensor.size)
                print("done ({:.1f} seconds)".format(time.time() - t1), flush=True)

            with open(sizes_file, 'wb') as f:
                pickle.dump(self.tensor_sizes, f)

            print("Processed input data: {:,d} characters loaded ({:.1f} seconds)".format(
                    self.tensor.size, time.time() - t0))
        else:
            # If the vocab file and sizes file already exist, load them.
            print("Loading vocab file...")
            self._load_vocab(vocab_file)
            print("Loading sizes file...")
            with open(sizes_file, 'rb') as f:
                self.tensor_sizes = pickle.load(f)
        self.tensor_batch_counts = [n // (self.batch_size * self.seq_length) for n in self.tensor_sizes]
        self.total_batch_count = sum(self.tensor_batch_counts)
        print("Total batch count: {:,d}".format(self.total_batch_count))

        self.tensor_index = -1

    def _preprocess_required(self, vocab_file, sizes_file, tensor_file_template, input_file_count):
        if not os.path.exists(vocab_file):
            print("No vocab file found. Preprocessing...")
            return True
        if not os.path.exists(sizes_file):
            print("No sizes file found. Preprocessing...")
            return True
        for i in range(input_file_count):
            if not os.path.exists(tensor_file_template.format(i)):
                print("Couldn't find {}. Preprocessing...".format(tensor_file_template.format(i)))
                return True
        return False

    def _get_input_file_list(self, data_dir):
        suffixes = ['.txt', '.bz2']
        input_file_list = []
        if os.path.isdir(data_dir):
            for walk_root, walk_dir, walk_files in os.walk(data_dir):
                for file_name in walk_files:
                    if file_name.startswith("."): continue
                    file_path = os.path.join(walk_root, file_name)
                    if file_path.endswith(suffixes[0]) or file_path.endswith(suffixes[1]):
                        input_file_list.append(file_path)
        else: raise ValueError("Not a directory: {}".format(data_dir))
        return sorted(input_file_list)

    def _save_vocab(self, vocab_file):
        self.chars = [chr(i) for i in range(128)]
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)
        print("Saved vocab (vocab size: {:,d})".format(self.vocab_size))

    def _load_vocab(self, vocab_file):
        # Load the character tuple (vocab.pkl) to self.chars.
        # Remember that it is in descending order of character frequency in the data.
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        # Use the character tuple to regenerate vocab_size and the vocab dictionary.
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

    def _preprocess(self, input_file, tensor_file):
        if input_file.endswith(".bz2"): file_reference = bz2.open(input_file, mode='rt')
        elif input_file.endswith(".txt"): file_reference = io.open(input_file, mode='rt')
        data = file_reference.read()
        file_reference.close()
        # Convert the entirety of the data file from characters to indices via the vocab dictionary.
        # How? map(function, iterable) returns a list of the output of the function
        # executed on each member of the iterable. E.g.:
        # [14, 2, 9, 2, 0, 6, 7, 0, ...]
        # np.array converts the list into a numpy array.
        self.tensor = np.array(list(map(self.vocab.get, data)))
        self.tensor = self.tensor[self.tensor != np.array(None)].astype(int) # Filter out None
        # Compress and save the numpy tensor array to data.npz.
        np.savez_compressed(tensor_file, tensor_data=self.tensor)

    def _load_preprocessed(self, tensor_index):
        self.reset_batch_pointer()
        if tensor_index == self.tensor_index:
            return
        print("loading tensor data file {}".format(tensor_index))
        tensor_file = self.tensor_file_template.format(tensor_index)
        # Load the data tensor file to self.tensor.
        with np.load(tensor_file) as loaded:
            self.tensor = loaded['tensor_data']
        self.tensor_index = tensor_index
        # Calculate the number of batches in the data. Each batch is batch_size x seq_length,
        # so this is just the input data size divided by that product, rounded down.
        self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)
        if self.tensor_batch_counts[tensor_index] != self.num_batches:
            print("Error in batch size! Expected {:,d}; found {:,d}".format(self.tensor_batch_counts[tensor_index],
                    self.num_batches))
        # Chop off the end of the data tensor so that the length of the data is a whole
        # multiple of the (batch_size x seq_length) product.
        # Do this with the slice operator on the numpy array.
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        # Construct two numpy arrays to represent input characters (xdata)
        # and target characters (ydata).
        # In training, we will feed in input characters one at a time, and optimize along
        # a loss function computed against the target characters.
        # (We do this with batch_size characters at a time, in parallel.)
        # Since this is a sequence prediction net, the target is just the input right-shifted
        # by 1.
        xdata = self.tensor
        ydata = np.copy(self.tensor) # Y-data starts as a copy of x-data.
        ydata[:-1] = xdata[1:] # Right-shift y-data by 1 using the numpy array slice syntax.
        # Replace the very last character of y-data with the first character of the input data.
        ydata[-1] = xdata[0]
        # Split our unidemnsional data array into distinct batches.
        # How? xdata.reshape(self.batch_size, -1) returns a 2D numpy tensor view
        # in which the first dimension is the batch index (from 0 to num_batches),
        # and the second dimension is the index of the character within the batch
        # (from 0 to (batch_size x seq_length)).
        # Within each batch, characters follow the same sequence as in the input data.
        # Then, np.split(that 2D numpy tensor, num_batches, 1) gives a list of numpy arrays.
        # Say batch_size = 4, seq_length = 5, and data is the following string:
        # "Here is a new string named data. It is a new string named data. It is named data."
        # We truncate the string to lop off the last period (so there are now 80 characters,
        # which is evenly divisible by 4 x 5). After xdata.reshape, we have:
        #
        # [[Here is a new string],
        #  [ named data. It is a],
        #  [ new string named da],
        #  [ta. It is named data]]
        #
        # After np.split, we have:
        # <[[Here ],   <[[is a ],   <[[new s],     <[[tring],
        #   [ name],     [d dat],     [a. It],       [ is a],
        #   [ new ],     [strin],     [g nam],       [ed da],
        #   [ta. I]]>,   [t is ]]>,   [named]]>,     [ data]]>
        #
        # where the first item of the list is the numpy array on the left.
        # Thus x_batches is a list of numpy arrays. The first dimension of each numpy array
        # is the batch number (from 0 to batch_size), and the second dimension is the
        # character index (from 0 to seq_length).
        #
        # These will be fed to the model one at a time sequentially.
        # State is preserved between sequential batches.
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        if self.tensor_index < 0:
            self._load_preprocessed(0)
        if self.pointer >= self.num_batches:
            self._load_preprocessed((self.tensor_index + 1) % self.input_file_count)
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def cue_batch_pointer_to_epoch_fraction(self, epoch_fraction):
        step_target = (epoch_fraction - int(epoch_fraction)) * self.total_batch_count
        self._cue_batch_pointer_to_step_count(step_target)

    def _cue_batch_pointer_to_step_count(self, step_target):
        for i, n in enumerate(self.tensor_batch_counts):
            if step_target < n:
                break
            step_target -= n
        self.pointer = n
        self.current_tensor_index = i
        self._load_preprocessed(i)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
#from tensorflow.contrib import rnn

from tensorflow.python.util.nest import flatten

import numpy as np

class PartitionedMultiRNNCell(rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    # Diagramn of a PartitionedMultiRNNCell net with three layers and three partitions per layer.
    # Each brick shape is a partition, which comprises one RNNCell of size partition_size.
    # The two tilde (~) characters indicate wrapping (i.e. the two halves are a single partition).
    # Like laying bricks, each layer is offset by half a partition width so that influence spreads
    # horizontally through subsequent layers, while avoiding the quadratic resource scaling of fully
    # connected layers with respect to layer width.

    #        output
    #  //////// \\\\\\\\
    # -------------------
    # |     |     |     |
    # -------------------
    # ~  |     |     |  ~
    # -------------------
    # |     |     |     |
    # -------------------
    #  \\\\\\\\ ////////
    #        input


    def __init__(self, cell_fn, partition_size=128, partitions=1, layers=2):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
            cell_fn: reference to RNNCell function to create each partition in each layer.
            partition_size: how many horizontal cells to include in each partition.
            partitions: how many horizontal partitions to include in each layer.
            layers: how many layers to include in the net.
        """
        super(PartitionedMultiRNNCell, self).__init__()

        self._cells = []
        for i in range(layers):
            self._cells.append([cell_fn(partition_size) for _ in range(partitions)])
        self._partitions = partitions

    @property
    def state_size(self):
        # Return a 2D tuple where each row is the partition's cell size repeated `partitions` times,
        # and there are `layers` rows of that.
        return tuple(((layer[0].state_size,) * len(layer)) for layer in self._cells)

    @property
    def output_size(self):
        # Return the output size of each partition in the last layer times the number of partitions per layer.
        return self._cells[-1][0].output_size * len(self._cells[-1])

    def zero_state(self, batch_size, dtype):
        # Return a 2D tuple of zero states matching the structure of state_size.
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return tuple(tuple(cell.zero_state(batch_size, dtype) for cell in layer) for layer in self._cells)

    def call(self, inputs, state):
        layer_input = inputs
        new_states = []
        for l, layer in enumerate(self._cells):
            # In between layers, offset the layer input by half of a partition width so that
            # activations can horizontally spread through subsequent layers.
            if l > 0:
                offset_width = layer[0].output_size // 2
                layer_input = tf.concat((layer_input[:, -offset_width:], layer_input[:, :-offset_width]),
                    axis=1, name='concat_offset_%d' % l)
            # Create a tuple of inputs by splitting the lower layer output into partitions.
            p_inputs = tf.split(layer_input, len(layer), axis=1, name='split_%d' % l)
            p_outputs = []
            p_states = []
            for p, p_inp in enumerate(p_inputs):
                with vs.variable_scope("cell_%d_%d" % (l, p)):
                    p_state = state[l][p]
                    cell = layer[p]
                    p_out, new_p_state = cell(p_inp, p_state)
                    p_outputs.append(p_out)
                    p_states.append(new_p_state)
            new_states.append(tuple(p_states))
            layer_input = tf.concat(p_outputs, axis=1, name='concat_%d' % l)
        new_states = tuple(new_states)
        return layer_input, new_states

def _rnn_state_placeholders(state):
    """Convert RNN state tensors to placeholders, reflecting the same nested tuple structure."""
    if isinstance(state, tf.compat.v1.nn.rnn_cell.LSTMStateTuple):
        c, h = state
        c = tf.placeholder(c.dtype, c.shape, c.op.name)
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return h
    else:
        structure = [_rnn_state_placeholders(x) for x in state]
        return tuple(structure)

class Model():
    def __init__(self, args, infer=False): # infer is set to true during sampling.
        self.args = args
        if infer:
            # Worry about one character at a time during sampling; no batching or BPTT.
            args.batch_size = 1
            args.seq_length = 1

        # Set cell_fn to the type of network cell we're creating -- RNN, GRU, LSTM or NAS.
        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # Create variables to track training progress.
        self.lr = tf.Variable(args.learning_rate, name="learning_rate", trainable=False)
        self.global_epoch_fraction = tf.Variable(0.0, name="global_epoch_fraction", trainable=False)
        self.global_seconds_elapsed = tf.Variable(0.0, name="global_seconds_elapsed", trainable=False)

        # Call tensorflow library tensorflow-master/tensorflow/python/ops/rnn_cell
        # to create a layer of block_size cells of the specified basic type (RNN/GRU/LSTM).
        # Use the same rnn_cell library to create a stack of these cells
        # of num_layers layers. Pass in a python list of these cells. 
        # cell = rnn_cell.MultiRNNCell([cell_fn(args.block_size) for _ in range(args.num_layers)])
        # cell = MyMultiRNNCell([cell_fn(args.block_size) for _ in range(args.num_layers)])
        cell = PartitionedMultiRNNCell(cell_fn, partitions=args.num_blocks,
            partition_size=args.block_size, layers=args.num_layers)

        # Create a TF placeholder node of 32-bit ints (NOT floats!),
        # of shape batch_size x seq_length. This shape matches the batches
        # (listed in x_batches and y_batches) constructed in create_batches in utils.py.
        # input_data will receive input batches.
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        self.zero_state = cell.zero_state(args.batch_size, tf.float32)

        self.initial_state = _rnn_state_placeholders(self.zero_state)
        self._flattened_initial_state = flatten(self.initial_state)

        layer_size = args.block_size * args.num_blocks

        # Scope our new variables to the scope identifier string "rnnlm".
        with tf.variable_scope('rnnlm'):
            # Create new variable softmax_w and softmax_b for output.
            # softmax_w is a weights matrix from the top layer of the model (of size layer_size)
            # to the vocabulary output (of size vocab_size).
            softmax_w = tf.get_variable("softmax_w", [layer_size, args.vocab_size])
            # softmax_b is a bias vector of the ouput characters (of size vocab_size).
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # Create new variable named 'embedding' to connect the character input to the base layer
            # of the RNN. Its role is the conceptual inverse of softmax_w.
            # It contains the trainable weights from the one-hot input vector to the lowest layer of RNN.
            embedding = tf.get_variable("embedding", [args.vocab_size, layer_size])
            # Create an embedding tensor with tf.nn.embedding_lookup(embedding, self.input_data).
            # This tensor has dimensions batch_size x seq_length x layer_size.
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # TODO: Check arguments parallel_iterations (default uses more memory and less time) and
        # swap_memory (default uses more memory but "minimal (or no) performance penalty")
        outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs,
                initial_state=self.initial_state, scope='rnnlm')
        # outputs has shape [batch_size, max_time, cell.output_size] because time_major == false.
        # Do we need to transpose the first two dimensions? (Answer: no, this ruins everything.)
        # outputs = tf.transpose(outputs, perm=[1, 0, 2])
        output = tf.reshape(outputs, [-1, layer_size])
        # Obtain logits node by applying output weights and biases to the output tensor.
        # Logits is a tensor of shape [(batch_size * seq_length) x vocab_size].
        # Recall that outputs is a 2D tensor of shape [(batch_size * seq_length) x layer_size],
        # and softmax_w is a 2D tensor of shape [layer_size x vocab_size].
        # The matrix product is therefore a new 2D tensor of [(batch_size * seq_length) x vocab_size].
        # In other words, that multiplication converts a loooong list of layer_size vectors
        # to a loooong list of vocab_size vectors.
        # Then add softmax_b (a single vocab-sized vector) to every row of that list.
        # That gives you the logits!
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        if infer:
            # Convert logits to probabilities. Probs isn't used during training! That node is never calculated.
            # Like logits, probs is a tensor of shape [(batch_size * seq_length) x vocab_size].
            # During sampling, this means it is of shape [1 x vocab_size].
            self.probs = tf.nn.softmax(self.logits)
        else:
            # Create a targets placeholder of shape batch_size x seq_length.
            # Targets will be what output is compared against to calculate loss.
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            # seq2seq.sequence_loss_by_example returns 1D float Tensor containing the log-perplexity
            # for each sequence. (Size is batch_size * seq_length.)
            # Targets are reshaped from a [batch_size x seq_length] tensor to a 1D tensor, of the following layout:
            #   target character (batch 0, seq 0)
            #   target character (batch 0, seq 1)
            #   ...
            #   target character (batch 0, seq seq_len-1)
            #   target character (batch 1, seq 0)
            #   ...
            # These targets are compared to the logits to generate loss.
            # Logits: instead of a list of character indices, it's a list of character index probability vectors.
            # seq2seq.sequence_loss_by_example will do the work of generating losses by comparing the one-hot vectors
            # implicitly represented by the target characters against the probability distrutions in logits.
            # It returns a 1D float tensor (a vector) where item i is the log-perplexity of
            # the comparison of the ith logit distribution to the ith one-hot target vector.

            loss = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]), logits=self.logits)

            # Cost is the arithmetic mean of the values of the loss tensor.
            # It is a single-element floating point tensor. This is what the optimizer seeks to minimize.
            self.cost = tf.reduce_mean(loss)
            # Create a tensorboard summary of our cost.
            tf.summary.scalar("cost", self.cost)

            tvars = tf.trainable_variables() # tvars is a python list of all trainable TF Variable objects.
            # tf.gradients returns a list of tensors of length len(tvars) where each tensor is sum(dy/dx).
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                     args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                'training_accuracy', dtype=tf.float32)
            # Use ADAM optimizer.
            # Zip creates a list of tuples, where each tuple is (variable tensor, gradient tensor).
            # Training op nudges the variables along the gradient, with the given learning rate, using the ADAM optimizer.
            # This is the op that a training session should be instructed to perform.
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            #self.train_op = optimizer.minimize(self.cost)
            self.summary_op = tf.summary.merge_all()

    def add_state_to_feed_dict(self, feed_dict, state):
        for i, tensor in enumerate(flatten(state)):
            feed_dict[self._flattened_initial_state[i]] = tensor

    def save_variables_list(self):
        # Return a list of the trainable variables created within the rnnlm model.
        # This consists of the two projection softmax variables (softmax_w and softmax_b),
        # embedding, and all of the weights and biases in the MultiRNNCell model.
        # Save only the trainable variables and the placeholders needed to resume training;
        # discard the rest, including optimizer state.
        save_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnnlm'))
        save_vars.update({self.lr, self.global_epoch_fraction, self.global_seconds_elapsed})
        return list(save_vars)

    def forward_model(self, sess, state, input_sample):
        '''Run a forward pass. Return the updated hidden state and the output probabilities.'''
        shaped_input = np.array([[input_sample]], np.float32)
        inputs = {self.input_data: shaped_input}
        self.add_state_to_feed_dict(inputs, state)
        [probs, state] = sess.run([self.probs, self.final_state], feed_dict=inputs)
        return probs[0], state

    def trainable_parameter_count(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


def main():
    assert sys.version_info >= (3, 3), \
    "Must be run in Python 3.3 or later. You are running {}".format(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='reddit-data',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='redditmodel',
                       help='directory for checkpointed models (load from here if one is already present)')
    parser.add_argument('--block_size', type=int, default=2048,
                       help='number of cells per block')
    parser.add_argument('--num_blocks', type=int, default=3,
                       help='number of blocks per layer')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, lstm or nas')
    parser.add_argument('--batch_size', type=int, default=40,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=40,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=5000,
                       help='save frequency')
    parser.add_argument('-f', type=str, type=str, default='/root/.local/share/jupyter/runtime/kernel-46850a28-6bfe-4077-8deb-8233d098e79f.json',
                       help='Google Colab Default')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.975,
                       help='how much to decay the learning rate')
    parser.add_argument('--decay_steps', type=int, default=100000,
                       help='how often to decay the learning rate')
    parser.add_argument('--set_learning_rate', type=float, default=-1,
                       help='reset learning rate to this value (if greater than zero)')
    args = parser.parse_args()
    train(args)

def train(args):
    # Create the data_loader object, which loads up all of our batches, vocab dictionary, etc.
    # from utils.py (and creates them if they don't already exist).
    # These files go in the data directory.
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    load_model = False
    if not os.path.exists(args.save_dir):
        print("Creating directory %s" % args.save_dir)
        os.mkdir(args.save_dir)
    elif (os.path.exists(os.path.join(args.save_dir, 'config.pkl'))):
        # Trained model already exists
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
                saved_args = pickle.load(f)
                args.block_size = saved_args.block_size
                args.num_blocks = saved_args.num_blocks
                args.num_layers = saved_args.num_layers
                args.model = saved_args.model
                print("Found a previous checkpoint. Overwriting model description arguments to:")
                print(" model: {}, block_size: {}, num_blocks: {}, num_layers: {}".format(
                    saved_args.model, saved_args.block_size, saved_args.num_blocks, saved_args.num_layers))
                load_model = True

    # Save all arguments to config.pkl in the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    # Save a tuple of the characters list and the vocab dictionary to chars_vocab.pkl in
    # the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)

    # Create the model!
    print("Building the model")
    model = Model(args)
    print("Total trainable parameters: {:,d}".format(model.trainable_parameter_count()))
    
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(model.save_variables_list(), max_to_keep=3)
        if (load_model):
            print("Loading saved parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
        global_epoch_fraction = sess.run(model.global_epoch_fraction)
        global_seconds_elapsed = sess.run(model.global_seconds_elapsed)
        if load_model: print("Resuming from global epoch fraction {:.3f},"
                " total trained time: {}, learning rate: {}".format(
                global_epoch_fraction,
                datetime.timedelta(seconds=float(global_seconds_elapsed)),
                sess.run(model.lr)))
        if (args.set_learning_rate > 0):
            sess.run(tf.assign(model.lr, args.set_learning_rate))
            print("Reset learning rate to {}".format(args.set_learning_rate))
        data_loader.cue_batch_pointer_to_epoch_fraction(global_epoch_fraction)
        initial_batch_step = int((global_epoch_fraction
                - int(global_epoch_fraction)) * data_loader.total_batch_count)
        epoch_range = (int(global_epoch_fraction),
                args.num_epochs + int(global_epoch_fraction))
        writer = tf.summary.FileWriter(args.save_dir, graph=tf.get_default_graph())
        outputs = [model.cost, model.final_state, model.train_op, model.summary_op]
        global_step = epoch_range[0] * data_loader.total_batch_count + initial_batch_step
        avg_loss = 0
        avg_steps = 0
        try:
            for e in range(*epoch_range):
                # e iterates through the training epochs.
                # Reset the model state, so it does not carry over from the end of the previous epoch.
                state = sess.run(model.zero_state)
                batch_range = (initial_batch_step, data_loader.total_batch_count)
                initial_batch_step = 0
                for b in range(*batch_range):
                    global_step += 1
                    if global_step % args.decay_steps == 0:
                        # Set the model.lr element of the model to track
                        # the appropriately decayed learning rate.
                        current_learning_rate = sess.run(model.lr)
                        current_learning_rate *= args.decay_rate
                        sess.run(tf.assign(model.lr, current_learning_rate))
                        print("Decayed learning rate to {}".format(current_learning_rate))
                    start = time.time()
                    # Pull the next batch inputs (x) and targets (y) from the data loader.
                    x, y = data_loader.next_batch()

                    # feed is a dictionary of variable references and respective values for initialization.
                    # Initialize the model's input data and target data from the batch,
                    # and initialize the model state to the final state from the previous batch, so that
                    # model state is accumulated and carried over between batches.
                    feed = {model.input_data: x, model.targets: y}
                    model.add_state_to_feed_dict(feed, state)
                    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                        'training_accuracy', dtype=tf.float32)

                    # Run the session! Specifically, tell TensorFlow to compute the graph to calculate
                    # the values of cost, final state, and the training op.
                    # Cost is used to monitor progress.
                    # Final state is used to carry over the state into the next batch.
                    # Training op is not used, but we want it to be calculated, since that calculation
                    # is what updates parameter states (i.e. that is where the training happens).
                    train_loss, state, _, summary = sess.run(outputs, feed)
                    elapsed = time.time() - start
                    global_seconds_elapsed += elapsed
                    writer.add_summary(summary, e * batch_range[1] + b + 1)
                    if avg_steps < 100: avg_steps += 1
                    avg_loss = 1 / avg_steps * train_loss + (1 - 1 / avg_steps) * avg_loss
                    print("{:,d} / {:,d} (epoch {:.3f} / {}), loss {:.3f} (avg {:.3f}), {:.3f}s" \
                        .format(b, batch_range[1], e + b / batch_range[1], epoch_range[1],
                            train_loss, avg_loss, elapsed))
                    print(training_acurracy.result)
                    # Every save_every batches, save the model to disk.
                    # By default, only the five most recent checkpoint files are kept.
                    if (e * batch_range[1] + b + 1) % args.save_every == 0 \
                            or (e == epoch_range[1] - 1 and b == batch_range[1] - 1):
                        save_model(sess, saver, model, args.save_dir, global_step,
                                data_loader.total_batch_count, global_seconds_elapsed)
        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            writer.flush()
            global_step = e * data_loader.total_batch_count + b
            save_model(sess, saver, model, args.save_dir, global_step,
                    data_loader.total_batch_count, global_seconds_elapsed)

def save_model(sess, saver, model, save_dir, global_step, steps_per_epoch, global_seconds_elapsed):
    global_epoch_fraction = float(global_step) / float(steps_per_epoch)
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
    print("Saving model to {} (epoch fraction {:.3f})...".format(checkpoint_path, global_epoch_fraction),
        end='', flush=True)
    sess.run(tf.assign(model.global_epoch_fraction, global_epoch_fraction))
    sess.run(tf.assign(model.global_seconds_elapsed, global_seconds_elapsed))
    saver.save(sess, checkpoint_path, global_step = global_step)
    print("\rSaved model to {} (epoch fraction {:.3f}).   ".format(checkpoint_path, global_epoch_fraction))

if __name__ == '__main__':
    main()
