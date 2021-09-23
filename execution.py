import training
import save_data as sd
from ansatz_classes import Ansatz_Pool, generate_network_parameters
import ansatz_classes as ac
from training_data import generate_training_data
import sys, os
import utils

import numpy as np # type: ignore

from typing import List, Optional, Union, NamedTuple, Any

def iteration_optimization(simulator: bool = True,
                        ansatz: Union[Ansatz_Pool] = Ansatz_Pool(name='dpnn',qnn_arch=[2,2]),
                        adapt_vqe: bool=False,
                        adapt_tol: float = 1e-1,
                        adapt_max_iter: int = 20,
                        shots: int=2**13,
                        epochs: int=1000,
                        optimize_method: str="Adam",
                        learning_rate: Optional[float]=1e-3,
                        analy_grad: bool=True,
                        order_of_derivative: int=2,
                        epsilon: float=0.25,
                        draw_circ: bool = True,
                        optimization_level: int=3,
                        num_training_pairs: Optional[int] = None,
                        device_name: Optional[str] = None,
                        simulation_method: str="matrix_product_state", 
                        load_params_from: Optional[str] = None) -> Any:
    """
    User-level function to set up a ansatz and train it on a given device.
    Default ansatz is the DQNN from "Training Quantum Neural Networks on NISQ devices" on a noise-free simulator.
    Creates a results folder in the output directory named by a timestamp. Saves all relevant parameters. Creates relevant figures.

    Args:
        simulator (bool, optional): If True, the ansatz is trained using a simulator instead of a real quantum device.
            It depends on the device_name if the simulator imitates the noise and qubit coupling of a real device
            or executes the corresponding quantum circuits noise-free.
            Defaults to True.
        name: the name of the ansatz:
            Defaults to the dqnn
        ansatz (Union[Ansatz_Pool]): A ansatz_pool object to be chosen.
            Defaults to name='dpnn' with Ansatz_Pool(qnn_arch=[2,2]).
        shots (int): Number of shots per circuit evaluation.
            Defaults to 2**13.
        epochs (int): Numer of epochs for training the given ansatz.
            Defaults to 100.
        optimize_method (str): Optimization method without the gradient algorithms are good.
            Defaults to 'COBYLA'.
        optimization_level (int, optional): Level with which the circuit is optimized by qiskit. Higher values mean better optimization,
            e.g. in terms of fewer 2-qubit gates and noise-adjusted qubit layout.
            Defaults to 3.
        device_name (Optional[str], optional): The device name of a IBMQ device.
            If a device is given either the noise properties and the qubit layout of this device are simulated (simulator==True)
            or the ansatz is trained on the real quantum device (simulator==False). Dependend on the IBMQ user account a long queuing time should be expected.
            The device must be accessible for the IBMQ user. The IBMQ user account can be configured in user_config.py. 
            If no device name is given a noise-free simulator is used (simulator==True) or the least busy device is used (simulator==False).
            Defaults to None.
        load_params_from (Optional[str], optional): The ansatz can be initialized with specific parameters which can be loaded from a local txt file.
            If no file is given the initial parameters are generated randomly.
            Defaults to None.
    Returns:
        Any: The final expectation values
    """

    # todo list
    if adapt_vqe and ansatz.name != 'ucc':
        print('Not support dqnn within the adapt_vqe !!!!')
        print('it will adapt_vqe=False to continue')
        adapt_vqe=False

    # Save timestamp and create results folder
    timestamp = sd.make_file_structure()

    # Fix device
    if not simulator:
        device_name = training.get_device(ansatz, simulator, device_name=device_name, do_calibration=False).name()
  
    if ansatz.name == 'dqnn' or ansatz.name == 'hardware':
        # Generate the initial parameters
        ansatz.update_params(params=load_params_from)

#        print('ansatz.params:',ansatz.params)
        # added noise to the model
##        ac.construct_noise_model(ansatz)

        # Pre-transpile parametrized circuits
        expectation=ac.construct_and_transpile_circuits(ansatz=ansatz, optimization_level=optimization_level, device_name=device_name, simulator=simulator,draw_circ=draw_circ)
        
    elif ansatz.name == 'ucc':
        # Generate the ucc ansatz and obtain the number of parameters
        if ansatz.excited_ranks == 'sd':
            ucc_operator_pool_fermOp, \
                ucc_operator_pool_qubitOp \
                = utils.generate_molecule_uccsd(ansatz.n_orb, ansatz.n_orb_occ, anti_hermitian=True)
        elif ansatz.excited_ranks == 'gsd':
            ucc_operator_pool_fermOp, \
                 ucc_operator_pool_qubitOp \
                 = utils.generate_molecule_uccgsd(ansatz.n_orb, ansatz.n_orb_occ, anti_hermitian=True)
        else:
            print('Not Support the type of excitation ranks !!!')
            sys.exit(0)

        if not adapt_vqe:
            ansatz.num_params =len(ucc_operator_pool_qubitOp)

#            print('ansatz.num_params:',ansatz.num_params)
            # Generate the initial parameters
            if load_params_from is None:
                ansatz.params=generate_network_parameters(num_params=ansatz.num_params)
            else: 
                ansatz.update_params(params=load_params_from)
#            print('ansatz.params:',ansatz.params)

            # Pre-transpile parametrized circuits
            expectation=ac.construct_and_transpile_circuits(ansatz=ansatz, optimization_level=optimization_level, 
                          ucc_operator_pool=ucc_operator_pool_qubitOp, device_name=device_name, simulator=simulator,draw_circ=draw_circ) 

    if not adapt_vqe:
        # evaluate the root of the possible variance
        if ansatz.tpb_grouping:
            reps=max(ansatz.ep_cont)*np.sqrt(ansatz.num_groups*(ansatz.nterms+ansatz.num_groups*
                  max(ansatz.group_member))/ansatz.nterms/shots)
        else:
            reps=max(ansatz.ep_cont)*np.sqrt(ansatz.nterms/shots)
    else:
        reps=0.0


    # Save all relevant parameters to an execution_info.txt
    sd.save_execution_info(simulator=simulator, ansatz=ansatz, shots=shots, epochs=epochs, optimize_method=optimize_method, 
        device_name=device_name, simulation_method=simulation_method, required_qubits=ansatz.required_qubits, 
        load_params_from=load_params_from if isinstance(load_params_from, str) else None, 
        num_entangle=ansatz.num_entangle if ansatz.name=='hardware' else None, 
        analy_grad=analy_grad, adapt_vqe=adapt_vqe, adapt_tol=adapt_tol if adapt_vqe else None, 
        adapt_max_iter=adapt_max_iter if adapt_vqe else None, 
        optimization_level=optimization_level, measurement_method=ansatz.meas_method,
        order_of_derivative=order_of_derivative if analy_grad else None, grouping=ansatz.tpb_grouping, 
        number_of_hamiltonian=ansatz.nterms, number_of_groups=ansatz.num_groups,MeanSquareError=reps)
    
    # the main vqe process
    if adapt_vqe:
        ep_final= training.adapt_vqe(ansatz=ansatz, analy_grad=analy_grad, 
                  ucc_operator_pool_qubitOp=ucc_operator_pool_qubitOp, adapt_tol=adapt_tol, adapt_max_iter=adapt_max_iter,
                  epochs=epochs, simulator=simulator, shots=shots, simulation_method=simulation_method,
                  optimize_method=optimize_method, learning_rate=learning_rate, device_name=str(device_name))
    else:
        ep_final= training.normal_vqe(ansatz=ansatz, expectation=expectation, analy_grad=analy_grad,
                  simulation_method=simulation_method, order_of_derivative=order_of_derivative, epochs=epochs, 
                  simulator=simulator, shots=shots, optimize_method=optimize_method, epsilon=epsilon,
                  learning_rate=learning_rate, device_name=str(device_name)) 

    print('--- TIMESTAMP: {} ---'.format(timestamp))

    return ep_final
