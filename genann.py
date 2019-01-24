"""
.. module:: genann

******
Genann
******

Genann is a minimal, well-tested library for training and using feedforward artificial neural networks (ANN) in C. Its primary focus is on being simple, fast, reliable, and hackable. It achieves this by providing only the necessary functions and little extra.

This module is a Python wrapper around the genann library hosted `here <https://github.com/codeplea/genann>`_ . It allows the execution of feedforward neural networks directly on the microcontroller. The training functions are not yet available.

   """


@native_c("_genann_run",[
    "csrc/genann_ifc.c",
    "csrc/genann/genann.c",
    "#csrc/math/exp.c",
    ],[],[])
def _run(ann,inputs,outputs):
    pass

@native_c("_genann_set_weight",["csrc/genann_ifc.c","csrc/genann/genann.c"],[],[])
def _set_weight(ann,i,w):
    pass

@native_c("_genann_create",["csrc/genann_ifc.c","csrc/genann/genann.c"],[],[])
def _create(inputs,outputs,nlayers,nhidden):
    pass

class ANN():
    """

=========
ANN Class
=========



.. class:: ANN()

This class is an implementations of the interface with the gennann C library.
Each istance of this class implements a separate neural network. The class is not thread safe,
multiple network must be protected by locks when executed.

    """
    def __init__(self):
        self.ann = None

    def create(self, inputs, outputs, nlayers, nhidden):
        """
.. method:: create(inputs,outputs,nlayers,nhidden)

    This method initializes a neural network with :samp:`inputs` inputs, :samp:`outputs` outputs and a number of hidden layers
    set by :samp:`nlayers` (that can be also 0). Each hidden layer, if present, contains exactly :samp:`nhidden` neurons.

        """
        self.ann = _create(inputs,outputs,nlayers,nhidden)
        self.output = [0.0]*outputs;
        self.outputs = outputs
        self.inputs = inputs
        self.layers = nlayers
        self.hidden = nhidden

    # def load(self, file, format="t"):
    #     line = file.readline()
    #     flds = line.split(" ")
    #     inputs = int(flds[0])
    #     layers = int(flds[1])
    #     hidden = int(flds[2])
    #     outputs = int(flds[3])
    #     self.create(inputs,outputs,layers,hidden)
    #     i = 0
    #     while line:
    #         line = file.readline()
    #         flds = line.split(" ")
    #         for ff in flds:
    #             f = float(ff)
    #             _set_weight(self.ann,i,f)
    #             i+=1

    # def write(self, file):
    #     pass

    def run(self, inputs):
        """
.. method:: run(inputs)

    This method executes the neural network using as input the array of float :samp:`inputs`.
    It returns an array of float representing the output layer.

        """
        _run(self.ann, inputs, self.output)
        return self.output

    # def train(self, inputs, outputs, learning_rate=3):
    #     pass

    def set_weights(self, weights):
        """
.. method:: set_weights(weights)

    This method initializes the weights of the ANN. It can be used to load a pre-trained set of weights or to 
    initializes the weights before training. Weights are represented by an array of floats :samp:`weights`.
        """
        i = 0
        for w in weights:
            # print("Setting weight",w)
            _set_weight(self.ann,i,w)
            i+=1
        # print("Exiting")

