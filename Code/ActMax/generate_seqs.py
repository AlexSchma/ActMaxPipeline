import torch
import numpy as np
from Code.Utils.Analysis_Utils import *
from Code.Utils.Data_Utils import *
import os
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

#import all classes/functions/data to create get_model function
from Code.Model_files.my_CNN import *

#define get_model function and other necessary variables
# i.e.
experiment_name = "Jordi_ActMax"
model_dict_path = "Data/Model_dicts/best_model.pt"

DNA_specs = [1500, 500, 500, 1500, "false"]
model_specs = {
        "n_ressidual_blocks": 4,
        "out_channels": 122,
        "kernel_size": 6,
        "max_pooling_kernel_size": 4,
        "dropout_rate": 0.5,
        "ffn_size_1": 128,
        "ffn_size_2": 64,
    }

n_sequences = 100
max_iters = 1000
early_stopping = True
patience = 50
sequence_length= np.sum(DNA_specs[:4]) + 20 if DNA_specs[4] == "false" else np.sum(DNA_specs[:4]) + 40 + 2000
#see below for more parameters

# define get_model function
def get_model(model_specs, DNA_specs, device, sequence_length):
    """
    Get the model
    """
    print(
        "Getting the model with required specifications:"
        f"\nDNA_specs: {DNA_specs}, \nmodel_specs: {model_specs}"
    )

    model = myCNN(
        sequence_length=sequence_length,
        n_labels=5,  # FIXED
        n_ressidual_blocks=model_specs["n_ressidual_blocks"],
        in_channels=4,  # FIXED
        out_channels=model_specs["out_channels"],
        kernel_size=model_specs["kernel_size"],
        max_pooling_kernel_size=model_specs["max_pooling_kernel_size"],
        dropout_rate=model_specs["dropout_rate"],
        ffn_size_1=model_specs["ffn_size_1"],
        ffn_size_2=model_specs["ffn_size_2"],
    )

    return model.to(device)


""" Define optimization objectives:
i.e. 
n_labels = 5
optimize one label at a time by setting the label to 1 and the rest to 0"""

optimization_objectives = {}
for i in range(5):
    optimization_objective_tensor = torch.zeros(5)
    optimization_objective_tensor[i] = 1
    optimization_objectives[f"label_{i}"] = optimization_objective_tensor
 
"""Also possible to optimize multiple labels at once
 or for specific levels of a label (has to add up to 1) 
 MSE between softmax output and optimization_objective_tensor
optimization_objectives = {
     "label_0_1": torch.tensor([0.5, 0.5, 0, 0, 0]),
     "label_all_equal": torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
}"""

#Further utils functions

def instance_normalize(logits):
    """
    Normalize the logits by subtracting the mean and dividing by the standard deviation.
    """
    mean = logits.mean(dim=2, keepdim=True)
    std = logits.std(dim=2, keepdim=True)
    return (logits - mean) / (std + 1e-5)

def reduce_parameter(initial_param, iteration, max_iters, end_param):
    """
    Linearly reduce a parameter from its initial value to its end value over a number of iterations.
    """
    return initial_param - (initial_param - end_param) * (iteration / max_iters)

def calculate_relative_hamming_distance(previous, current):
    """
    Calculate the relative Hamming distance between two sets of sequences.
    """
    prev_indices = previous.argmax(dim=1)
    curr_indices = current.argmax(dim=1)
    differences = (prev_indices != curr_indices).float()  
    relative_distances = differences.sum(dim=-1) / differences.shape[-1]  
    return relative_distances

def entropy_loss_func(pwm):
    """
    Calculate the entropy loss for a given probability weight matrix (PWM).
    """
    pwm = torch.clamp(pwm, min=1e-9, max=1 - 1e-9)
    entropy = -pwm * torch.log2(pwm)
    entropy = entropy.sum(dim=1)
    mean_entropy = entropy.mean(dim=1)
    return mean_entropy.mean()

def target_entropy_mse(pwm, target_bits):
    """
    Calculate the mean squared error (MSE) between the entropy of the PWM and the target entropy.
    """
    pwm_clipped = torch.clamp(pwm, min=1e-8, max=1.0 - 1e-8)
    entropy = pwm_clipped * -torch.log(pwm_clipped) / torch.log(torch.tensor(2.0))
    entropy = torch.sum(entropy, dim=1)
    conservation = 2.0 - entropy
    mse = torch.mean((conservation - target_bits)**2)
    return mse

def argmax_to_nucleotide(argmax_sequences, argmax_mapping={0:"A",1:"C",2:"T",3:"G"}):
    """
    Convert argmax indices to nucleotide sequences using a provided mapping.
    """
    nuc_seqs = []
    for argmax_seq in argmax_sequences:
        nuc_seqs.append("".join([argmax_mapping[int(integer)] for integer in argmax_seq]))
    return nuc_seqs

def visualize_and_save(losses, hamming_distances, results_path, cond, show=False):
    """
    Visualize and save the loss and average relative Hamming distance over iterations.
    """
    save_path = os.path.join(results_path, f"{cond}_loss_vs_hamming.png")

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(len(losses)), losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average Relative Hamming Distance', color=color)
    ax2.plot(range(len(hamming_distances)), hamming_distances, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Loss and Average Relative Hamming Distance over Iterations')
    plt.savefig(save_path)
    if show:
        plt.show()

def train(model,save_hyperparam,early_stopping, patience, optimization_objective_tensor,
                                optimization_objective_label, model_dict_path,results_path, input_tensor_shape, device, lr, initial_tau, final_tau, initial_entropy_weight, final_entropy_weight, target_bits, max_iters, verbose, change_tau, use_custom_entropy_loss, use_fast_seq_entropy_loss):
    
    #create directory for results
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    model.to(device)
    print("Model created")
    model_state_dict = torch.load(model_dict_path)
    print("Model state dict loaded")
    model.load_state_dict(model_state_dict)
    model.to(device)
    print("Model loaded from checkpoint")
    model.eval()

    input_tensor = torch.randn(input_tensor_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_tensor], lr=lr)
    optimization_objective_tensor = optimization_objective_tensor.to(device)

    losses = []
    hamming_distances = []
    count = 0
    sample_cache = torch.zeros(input_tensor_shape, device=device)

    assert not (use_custom_entropy_loss and use_fast_seq_entropy_loss), "Choose only one entropy loss function"

    early_stopping_enabled = early_stopping  
    early_stopping_patience = patience 

    for i in range(max_iters):
        optimizer.zero_grad()
        if change_tau:
            tau = reduce_parameter(initial_tau, i, max_iters, final_tau)
        else:
            tau = initial_tau
        normalized_logits = instance_normalize(input_tensor)
        pwm = F.softmax(normalized_logits, dim=1)
        samples = F.gumbel_softmax(normalized_logits, tau=tau, hard=True, dim=1)
        if i > 0:
            hamming_distance = calculate_relative_hamming_distance(sample_cache, samples)
            hamming_distances.append(hamming_distance.mean().item())
            
            if hamming_distance.mean().item() < best_hamming_distance:
                best_hamming_distance = hamming_distance.mean().item()
                count = 0  # Reset the counter if there's an improvement
            else:
                count += 1  # Increment the counter if there's no improvement

            if count == early_stopping_patience and early_stopping_enabled:
                print(f"Early stopping at iteration {i} due to no improvement in hamming distance for {early_stopping_patience} epochs.")
                break
        else:
            best_hamming_distance = float('inf')  # Initialize the best hamming distance for the first iteration
            count = 0
            
        sample_cache = samples.clone()
        
        output = model(samples)
        output = torch.softmax(output, dim=1)
        target = optimization_objective_tensor.repeat(input_tensor_shape[0], 1)

        # Compute MSE Loss (function averages over batch automatically)
        target_loss = F.mse_loss(output, target)

        if use_custom_entropy_loss:
            entropy_weight = reduce_parameter(initial_entropy_weight, i, max_iters, final_entropy_weight)
            loss = entropy_weight * entropy_loss_func(pwm) + target_loss
        elif use_fast_seq_entropy_loss:
            entropy_loss = target_entropy_mse(pwm, target_bits=target_bits)
            loss = target_loss + entropy_loss
        else:
            loss = target_loss
        
        if verbose and i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")
            print(f"Output: {output.mean(dim=0)}")
        if i > 0:
            losses.append(loss.item())

        loss.backward()
        optimizer.step()

    optimized_inputs = input_tensor.detach().cpu()
    argmax_optimized_input = np.argmax(optimized_inputs, 1)
    
    nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    visualize_and_save(losses, hamming_distances, results_path,optimization_objective_label)
    # Save nuc_seqs as fasta
    fasta_path = os.path.join(results_path, f"{optimization_objective_label}.fasta")
    with open(fasta_path, "w") as fasta_file:
        for i, seq in enumerate(nuc_seqs):
            fasta_file.write(f">Sequence_{i+1}\n")
            fasta_file.write(f"{seq}\n")

    # Save nuc_seqs as txt
    txt_path = os.path.join(results_path, f"{optimization_objective_label}.txt")
    with open(txt_path, "w") as txt_file:
        for i, seq in enumerate(nuc_seqs):
            txt_file.write(f"Sequence {i+1}: {seq}\n")

    # Save nuc_seqs as numpy file
    np_path = os.path.join(results_path, f"{optimization_objective_label}.npy")
    np.save(np_path, nuc_seqs)


    if save_hyperparam:
        hyperparam_path = os.path.join(results_path, "hyperparameters.json")
        hyperparameters = {
            "model_dict_path": model_dict_path,
            "results_path": results_path,
            "input_tensor_shape": list(input_tensor_shape),
            "device": str(device),
            "lr": lr,
            "initial_tau": initial_tau,
            "final_tau": final_tau,
            "initial_entropy_weight": initial_entropy_weight,
            "final_entropy_weight": final_entropy_weight,
            "target_bits": target_bits,
            "max_iters": max_iters,
            "verbose": verbose,
            "change_tau": change_tau,
            "use_custom_entropy_loss": use_custom_entropy_loss,
            "use_fast_seq_entropy_loss": use_fast_seq_entropy_loss
        }

        with open(hyperparam_path, 'w') as json_file:            
            json.dump(hyperparameters, json_file)
    return nuc_seqs, losses, hamming_distances


if __name__ == '__main__':
    print("Imported all necessary modules")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = get_model(model_specs, DNA_specs, device,sequence_length)
    print(model)
    print("Model created")
  
    for iteration, (optimization_objective_label,optimization_objective_tensor) in enumerate(optimization_objectives.items()):
        print(f"Activation Maximization: {optimization_objective_label}")
        print(f"Optimizing for: {optimization_objective_tensor}")
        if iteration == 0:
            save_hyperparam = True
        nuc_seqs,loss, ham_dist = train(model=model,
                                save_hyperparam = save_hyperparam, early_stopping=early_stopping, patience=patience,
                                optimization_objective_tensor=optimization_objective_tensor,
                                optimization_objective_label=optimization_objective_label,
                                model_dict_path=model_dict_path,
                                results_path=f"Data/Results/{experiment_name}",
                                input_tensor_shape=(n_sequences,4,int(sequence_length)),
                                device=device, 
                                lr=0.1,  
                                initial_tau = 1.0, 
                                final_tau = 0.1, 
                                initial_entropy_weight = 0.01, 
                                final_entropy_weight = 0.001, 
                                target_bits = 0.5, 
                                max_iters = max_iters, 
                                verbose = True, 
                                change_tau = True, 
                                use_custom_entropy_loss = False, 
                                use_fast_seq_entropy_loss = False)
    
    

    

    
