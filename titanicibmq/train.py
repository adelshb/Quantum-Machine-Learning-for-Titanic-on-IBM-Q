# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi, https://github.com/adelshb
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
""" Implementation of a QNN for the Titanic dataset

Training script
"""

from argparse import ArgumentParser

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

from titanicibmq.titanic_data import *

_available_optimizers = {
    "cobyla": COBYLA
}

_available_ansatz = {
    "realamplitudes": RealAmplitudes
}

_available_feature_maps = {
    "zzfeaturemap": ZZFeatureMap
}

def main(args):

    # Load the data
    data, __ = titanic()
    X_train, y_train, X_test, y_test = parse_data_train_vqc(data, split_ratio=args.split_ratio)

    quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'), shots=100)
    optimizer = COBYLA(maxiter=100)
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
    ansatz = RealAmplitudes(num_qubits=feature_map._num_qubits, reps=1)

    qc = QuantumCircuit(feature_map._num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    qnn = CircuitQNN(circuit= qc, 
                    input_params=feature_map.parameters, 
                    weight_params=ansatz.parameters, 
                    sparse=False, 
                    sampling=False, 
                    interpret=parity, 
                    output_shape=len(np.unique(y_train, axis=0)), 
                    gradient=None, 
                    quantum_instance=quantum_instance)

    cc = NeuralNetworkClassifier(neural_network=qnn,
                                    optimizer=optimizer)

    # Train the model
    cc.fit(X_train, y_train)

    # Model accuracy
    acc_train = cc.score(X_train, y_train)
    acc_test = cc.score(X_test, y_test)
    print("Accuracy on training dataset: {}.".format(acc_train))
    print("Accuracy on testing dataset: {}.".format(acc_test))

def parity(x):
    return '{:b}'.format(x).count('1') % 2

if __name__ == "__main__":
    parser = ArgumentParser()

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="cobyla", choices=_available_optimizers)
    parser.add_argument("--max_iter", type=int, default=1000)

    # Ansatz
    parser.add_argument("--ansatz", type=str, default="realamplitudes", choices=_available_ansatz)
    parser.add_argument("--a_reps", type=int, default=3)

    # Feature Map
    parser.add_argument("--feature_map", type=str, default="zzfeaturemap", choices=_available_feature_maps)
    parser.add_argument("--feature_dim", type=int, default=2)
    parser.add_argument("--f_reps", type=int, default=1)

    # Backend
    parser.add_argument("--backend", type=str, default="qasm_simulator")
    parser.add_argument("--shots", type=int, default=1024)

    # Data
    parser.add_argument("--split_ratio", type=int, default=0.2)

    args = parser.parse_args()
    main(args)