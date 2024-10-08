import random
import sys
sys.path.append("./")
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
#creates synthethic sequence/label pair where half of the random DNA sequences created have one motif inserted at a random position (label =1, else 0)
def generate_sequence(length=100, motifs=None):
    bases = "ACGT"
    sequence = ''.join(random.choice(bases) for _ in range(length))
    if motifs and random.random() > 0.5:  # Insert motifs randomly for one class
        motif = random.choice(motifs)
        start = random.randint(0, length - len(motif))
        # Replace a segment of the original sequence with the motif to keep the length constant
        sequence = sequence[:start] + motif + sequence[start + len(motif):]
        label = 1
    else:
        label = 0
    return sequence, label
#creates synthethic sequence/label/condition triplet.
#Each motif corresponds to one condition. Half of the sequences created have one motif inserted which corresponds to their condition (label =1), 25% of created sequences are completely random (label = 0) and 25% have a motif inserted that does not match the condition (label = 0)
def generate_sequence_conditional(length=100, motifs=None):
    bases = "ACGT"
    sequence = ''.join(random.choice(bases) for _ in range(length))
    n_motifs = len(motifs)
    #positive class
    if random.random() > 0.5:
        cond_mot = random.randint(0,n_motifs-1)
        motif = motifs[cond_mot]
        start = random.randint(0, length - len(motif))
        # Replace a segment of the original sequence with the motif to keep the length constant
        sequence = sequence[:start] + motif + sequence[start + len(motif):]
        label = 1
        condition = cond_mot
    else:
        if random.random() >0.5:
            label = 0
            condition = random.randint(0,n_motifs-1)
        else:
            motif_idx, condition = tuple(random.sample(range(0, n_motifs), 2))
            motif = motifs[motif_idx]
            start = random.randint(0, length - len(motif))
            # Replace a segment of the original sequence with the motif to keep the length constant
            sequence = sequence[:start] + motif + sequence[start + len(motif):]
            label = 0
    return sequence, label, condition
def one_hot_encode(sequences):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 'G': [0, 0, 0, 1]} #jordi mapping
    one_hot_encoded = np.array([[mapping[nucleotide] for nucleotide in sequence] for sequence in sequences])
    return one_hot_encoded

def decode_from_argmax(indices_sequences):
    # Mapping from index to nucleotides
    index_to_nucleotide = {0: 'A', 1: 'C', 2: 'T', 3: 'G'} #jordi mapping
    
    # Decode each sequence of indices
    decoded_sequences = []
    for sequence in indices_sequences:
        # Map each index in the sequence to its corresponding nucleotide
        decoded_sequence = ''.join([index_to_nucleotide[index] for index in sequence])
        decoded_sequences.append(decoded_sequence)
    
    return decoded_sequences
def argmax_to_nucleotide(argmax_sequences):
    argmax_mapping = {0:"A",1:"C",2:"T",3:"G"} #jordi mapping
    nuc_seqs = []
    for argmax_seq in argmax_sequences:
        nuc_seqs.append("".join([argmax_mapping[int(integer)] for integer in argmax_seq]))
    return nuc_seqs

def create_synth_test_data(motifs=["TTCCCTCG", "CTACCTCC", "TGCAGTGC"], n_synth=6000, batch_size=128):
    sequences, labels, conditions = zip(*(generate_sequence_conditional(motifs=motifs) for _ in range(n_synth)))
    one_hot_sequences = one_hot_encode(sequences)

    X_test = np.array(one_hot_sequences)
    y_test = np.array(labels)
    z_test = np.array(conditions)

    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test),torch.Tensor(z_test))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return test_loader, X_test, y_test, conditions

def create_synth_data(motifs,n_synth = 30000, test_size = 0.4, val_size = 0.5, batch_size = 128):
     
    # Generate synthetic data
    sequences, labels, condition = zip(*(generate_sequence_conditional(motifs=motifs) for _ in range(30000))) 
    one_hot_sequences = one_hot_encode(sequences)

    # Split data into training, validation, and test sets
    X = np.array(one_hot_sequences)
    y = np.array(labels)
    z = np.array(condition)
    X_train, X_temp, y_train, y_temp, z_train,z_temp = train_test_split(X, y,z, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test, z_val, z_test = train_test_split(X_temp, y_temp,z_temp, test_size=val_size, random_state=42)
    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train),torch.Tensor(z_train))
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val),torch.Tensor(z_val))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test),torch.Tensor(z_test))

    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)


    return train_loader, val_loader, test_loader

def create_synth_data_uncond(motifs,n_synth = 30000, test_size = 0.4, val_size = 0.5, batch_size = 128):
     
    sequences, labels = zip(*(generate_sequence(motifs=motifs) for _ in range(n_synth)))  
    one_hot_sequences = one_hot_encode(sequences)


    X = np.array(one_hot_sequences)
    y = np.array(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # Create DataLoaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader