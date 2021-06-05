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
""" Implementation of a VQC for the Titanic dataset

Training script
"""

from argparse import ArgumentParser

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms.classifiers import VQC
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

    # Initialize Quantum Backend
    quantum_instance = QuantumInstance(Aer.get_backend(args.backend), shots = args.shots)

    # Construct VQC
    optimizer = _available_optimizers.get(args.optimizer)(maxiter=args.max_iter)
    feature_map = _available_feature_maps.get(args.feature_map)(feature_dimension=args.feature_dim, reps=args.f_reps)
    ansatz = _available_ansatz.get(args.ansatz)(num_qubits=feature_map.num_qubits, reps=args.a_reps)
    # optimizer = COBYLA(maxiter=args.max_iter)
    # feature_map = ZZFeatureMap(feature_dimension=args.feature_dim, reps=args.f_reps)
    # ansatz = RealAmplitudes(num_qubits=feature_map.num_qubits, reps=args.a_reps)

    vqc = VQC(feature_map=feature_map, 
                ansatz=ansatz, 
                loss='cross_entropy', 
                optimizer=optimizer, 
                warm_start=False, 
                quantum_instance=quantum_instance)

    # Train the model
    vqc.fit(X_train, y_train)

    # Model accuracy
    acc_train = vqc.score(X_train, y_train)
    acc_test = vqc.score(X_test, y_test)
    print("Accuracy on training dataset: {}.".format(acc_train))
    print("Accuracy on testing dataset: {}.".format(acc_test))


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