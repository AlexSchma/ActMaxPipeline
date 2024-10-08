import torch
import numpy as np
from Analysis_Utils import *
from Data_Utils import *
from FunProseModels import *
import os
import json

import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
torch.set_float32_matmul_precision('high')




def instance_normalize(logits):
    mean = logits.mean(dim=2, keepdim=True)
    std = logits.std(dim=2, keepdim=True)
    return (logits - mean) / (std + 1e-5)

def reduce_parameter(initial_param, iteration, max_iters,end_param):
    return initial_param - (initial_param - end_param) * (iteration / max_iters)

def calculate_relative_hamming_distance(previous, current):
    prev_indices = previous.argmax(dim=1)
    curr_indices = current.argmax(dim=1)
    differences = (prev_indices != curr_indices).float()  
    relative_distances = differences.sum(dim=-1) / differences.shape[-1]  
    return relative_distances


def entropy_loss_func(pwm):
    pwm = torch.clamp(pwm, min=1e-9, max=1 - 1e-9)
    entropy = -pwm * torch.log2(pwm)
    entropy = entropy.sum(dim=1)
    mean_entropy = entropy.mean(dim=1)
    return mean_entropy.mean()


def target_entropy_mse(pwm, target_bits):
    pwm_clipped = torch.clamp(pwm, min=1e-8, max=1.0 - 1e-8)
    entropy = pwm_clipped * -torch.log(pwm_clipped) / torch.log(torch.tensor(2.0))
    entropy = torch.sum(entropy, dim=1)
    conservation = 2.0 - entropy
    mse = torch.mean((conservation - target_bits)**2)
    return mse


def visualize_and_save(main_losses, adv_losses, total_losses, hamming_distances, results_path, cond,opt_goal, show=False):
    save_path = os.path.join(results_path, f"{cond}_{opt_goal}_loss_vs_hamming.png")

    fig, ax1 = plt.subplots()

    # Plotting Losses
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    
    # Main loss plot
    ax1.plot(range(len(main_losses)), main_losses, color='tab:red', label=f'Loss {cond}')
    
    # Adversarial losses plot
    for key, adv_loss in adv_losses.items():
        ax1.plot(range(len(adv_loss)), adv_loss, label=f'Adv Loss {key}', linestyle=':', alpha=0.7)
    
    # # Total loss plot
    # ax1.plot(range(len(total_losses)), total_losses, color='tab:orange', label='Total Loss')
    # ax1.tick_params(axis='y')
    
    # Adding the second y-axis for Hamming Distance
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Average Relative Hamming Distance', color='tab:brown')
    ax2.plot(range(len(hamming_distances)), hamming_distances, color='tab:brown', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:brown')

    # Create a combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    fig.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig(save_path)
    
    if show:
        plt.show()

def save_losses_and_hamming(losses, hamming_distances, results_path, cond,opt_goal):
    # Prepare the dictionary structure
    data = {
        "total_losses": losses[0],
        "adv_losses": losses[1],  # This can be a dictionary directly
        "main_losses": losses[2],
        "hamming_distances": hamming_distances,
    }
    
    # Save as a JSON file
    save_path = os.path.join(results_path, f"{cond}_{opt_goal}_losses_hamming.json")
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def train(model, model_dict_path, adv_cond_tensor_dict, iteration, cond, 
          results_path, input_tensor_shape, device, lr, cond_tensor, initial_tau, 
          final_tau, initial_entropy_weight, final_entropy_weight, target_bits, max_iters, 
          verbose, change_tau, use_custom_entropy_loss, use_fast_seq_entropy_loss,target,
          opt_goal):
    model.to(device)
    model = torch.compile(model)
    
    print("Model created")
    model_state_dict = torch.load(model_dict_path)
    print("Model state dict loaded")
    model.load_state_dict(model_state_dict)
    model.to(device)
    print("Model loaded from checkpoint")
    model.eval()

    input_tensor = torch.randn(input_tensor_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    main_losses = []
    total_losses = []
    adv_losses = {k: [] for k in adv_cond_tensor_dict.keys()}
    hamming_distances = []
    count = 0
    sample_cache = torch.zeros(input_tensor_shape, device=device)

    assert not (use_custom_entropy_loss and use_fast_seq_entropy_loss), "Choose only one entropy loss function"

    # Helper function to handle conditional loss calculations
    def compute_loss(target_loss, use_custom_entropy_loss, use_fast_seq_entropy_loss, pwm, i, max_iters):
        if use_custom_entropy_loss:
            entropy_weight = reduce_parameter(initial_entropy_weight, i, max_iters, final_entropy_weight)
            return entropy_weight * entropy_loss_func(pwm) + target_loss
        elif use_fast_seq_entropy_loss:
            entropy_loss = target_entropy_mse(pwm, target_bits=target_bits)
            return target_loss + entropy_loss
        else:
            return target_loss
    with tqdm(total=max_iters) as pbar:
        for i in range(max_iters):
            total_loss = 0
            optimizer.zero_grad()  # Zero gradients once at the start of the iteration

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

            if torch.allclose(samples, sample_cache):
                count += 1
                if count == 10:
                    break
            else:
                count = 0
                sample_cache = samples.clone()
            
            total_loss = 0
            thresholds = {'main': -0.8}  # Define a threshold for the main loss

            # Main condition tensor
            output = model(samples, cond_tensor)
            output = torch.softmax(output, dim=1)
            main_target_loss = -torch.mean(output[:, target])
            main_loss = compute_loss(main_target_loss, use_custom_entropy_loss, use_fast_seq_entropy_loss, pwm, i, max_iters)

            adv_losses_iteration = []
            for adv_cond, tensor in adv_cond_tensor_dict.items():
                adv_output = model(samples, tensor)
                adv_output = torch.softmax(adv_output, dim=1)
                adv_target_loss = -torch.mean(adv_output[:, 1])
                adv_loss = compute_loss(adv_target_loss, use_custom_entropy_loss, use_fast_seq_entropy_loss, pwm, i, max_iters)
                adv_losses_iteration.append(adv_loss)
                if i > 0:
                    adv_losses[adv_cond].append(adv_loss.item())
            # Lexicographic prioritization
            # Check if main loss is below the threshold before considering adversarial losses
            if main_target_loss > thresholds['main']:
                total_loss = main_loss
            else:
                # Main loss is acceptable; proceed with adversarial losses

                # Sum up all adversarial losses
                total_loss = sum(adv_losses_iteration)

            

            total_loss.backward()
            if verbose and i % 10 == 0:
                print(f"Iteration {i}, Main Loss: {main_loss.item()}")
                print(f"Output: {output.mean(dim=0)}")

            if i > 0:
                main_losses.append(main_loss.item())
                total_losses.append(total_loss.mean().item())
            optimizer.step()  # Update parameters once after accumulating all gradients

            if i >0:
                loss_dict = {
                    "Main Loss": main_losses[-1],
                    "Total Loss": total_losses[-1],
                }

                for adv_cond, adv_loss_list in adv_losses.items():
                    loss_dict[f"Adv Loss {adv_cond}"] = adv_loss_list[-1]  # Latest adversarial loss

                pbar.set_postfix(loss_dict)
                pbar.update(1)


    optimized_inputs = input_tensor.detach().cpu()
    argmax_optimized_input = np.argmax(optimized_inputs, 1)
    nuc_seqs = argmax_to_nucleotide(argmax_optimized_input)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    visualize_and_save(main_losses, adv_losses, total_losses, hamming_distances, results_path, cond,opt_goal)
    save_losses_and_hamming((total_losses, adv_losses, main_losses), hamming_distances, results_path, cond, opt_goal)

    # Save nuc_seqs as fasta
    fasta_path = os.path.join(results_path, f"{cond}_{opt_goal}.fasta")
    with open(fasta_path, "w") as fasta_file:
        for i, seq in enumerate(nuc_seqs):
            fasta_file.write(f">Sequence_{i+1}\n")
            fasta_file.write(f"{seq}\n")

    # Save nuc_seqs as txt
    txt_path = os.path.join(results_path, f"{cond}_{opt_goal}.txt")
    with open(txt_path, "w") as txt_file:
        for i, seq in enumerate(nuc_seqs):
            txt_file.write(f"Sequence {i+1}: {seq}\n")

    # Save nuc_seqs as numpy file
    np_path = os.path.join(results_path, f"{cond}_{opt_goal}.npy")
    np.save(np_path, nuc_seqs)

    if iteration == 0:
        hyperparameters = {
            "model_dict_path": model_dict_path,
            "results_path": results_path,
            "input_tensor_shape": input_tensor_shape,
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

        with open(os.path.join(results_path, "hyperparameters.json"), "w") as json_file:
            json.dump(hyperparameters, json_file)

    return nuc_seqs, (total_losses, adv_losses, main_losses), hamming_distances


if __name__ == '__main__':
    print("Imported all necessary modules")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model = get_funprose_multi_class()
    print("Model created")
    #cond_tensor = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]], device=device).float()

    # Dictionary of condition embeddings
    cond_tensor_dict = {
        "ABA": [1, 0, 0],
        "SA": [0, 1, 0],
        "JA": [0, 0, 1],
        "ABA_JA": [1, 0, 1],
        "SA_JA": [0, 1, 1],
    }
    # cond_tensor_dict = {
    #     "ABA": [1, 0, 0],
    #     "SA": [0, 1, 0],
    #     "JA": [0, 0, 1]
    # }

    n_sequences = 100
    nuc_seq_dict = {}
    experiment_name = "DO8_108_endmodel2"
    print(f"Experiment Name: {experiment_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over conditions and embeddings
    opt_goal_dict = {"up": 2, "down": 0}
    #opt_goal_dict = {"up": 2}
    for opt_goal, target in opt_goal_dict.items():
        for i, (cond, embedding) in enumerate(cond_tensor_dict.items()):
            # Create the primary condition tensor
            cond_tensor = torch.tensor([embedding] * n_sequences, device=device).float()

            # Determine alternative conditions by excluding the current condition
            alt_conds = {k: v for k, v in cond_tensor_dict.items() if k != cond}

            # List to hold alternative condition tensors
            adv_cond_tensor_dict = {}

            # Generate alternative condition tensors
            for alt_cond, alt_embedding in alt_conds.items():
                adv_cond_tensor = torch.tensor([alt_embedding] * n_sequences, device=device).float()
                adv_cond_tensor_dict[alt_cond] = adv_cond_tensor

            nuc_seq_dict[cond], loss, ham_dist = train(model=model,
                                                    model_dict_path="models/multiclass_standard_metrics_0.8_dropout_108ep4_model.pth",
                                                    adv_cond_tensor_dict=adv_cond_tensor_dict,
                                                    iteration=i,
                                                    cond=cond,
                                                    results_path=f"generated_sequences/{experiment_name}",
                                                    input_tensor_shape=(n_sequences, 4, 4001),
                                                    device=device,
                                                    lr=0.1,
                                                    cond_tensor=cond_tensor,
                                                    initial_tau=1.0,
                                                    final_tau=0.1,
                                                    initial_entropy_weight=0.01,
                                                    final_entropy_weight=0.001,
                                                    target_bits=0.5,
                                                    max_iters=300,
                                                    verbose=False,
                                                    change_tau=True,
                                                    use_custom_entropy_loss=False,
                                                    use_fast_seq_entropy_loss=False,
                                                    target=target,
                                                    opt_goal=opt_goal)
            
