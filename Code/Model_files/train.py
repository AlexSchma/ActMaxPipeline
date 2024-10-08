import sys

sys.path.append(".")
import os
# from Code.Training.Data_Handling.CustomDataset import get_dataloader
# from Code.Training.Data_Handling.GenerateDataSplits import DataHandler
from Code.Model_files.my_CNN import myCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

## Let's define a class to handle the training in any scenarion, say hyperparameter tuning or normal training.
# This trainer class will take all the details as arguments and have all the required methods to do training, hyperparameter tuning and testing.


class Trainer:
    def __init__(
        self,
        model_specs: dict,
        DNA_specs: list,
        # train_proportion: float,
        # val_proportion: float,
        # train_specs: dict,
        # storage_path: str,
        fws_random_state=43123,
    ):
        """

        Args:
        model_specs: dict
            Dictionary containing the specifications of the model to be trained.
            Example:
            model_specs = {
                "n_ressidual_blocks": 3,
                "out_channels": 64,
                "kernel_size": 3,
                "max_pooling_kernel_size": 2,
                "dropout_rate": 0.5,
                "ffn_size_1": 128,
                "ffn_size_2": 64,
            }

        DNA_specs: np.array
            Array containing the specifications of the DNA sequences.
            Example:
            DNA_specs = [2000, 1000, 1000, 2000, "false"] for upstream, downstream TSS and TTS and either if introns are used or not.

        train_proportion: float
            Proportion of the data to be used for training.

        val_proportion: float
            Proportion of the training data to be used for validation.

        fws_random_state: int
            Random state for the family wise splitting.

        train_specs: dict
            Dictionary containing the specifications of the training.
            Example:
            train_specs = {
                "lr": 0.001,
                "weight_decay": 0.0001,
                "n_epochs": 100, # FIXED
                "batch_size": 64, # FIXED
            }

        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_specs = model_specs
        self.DNA_specs = DNA_specs
        assert DNA_specs[4] in ["true", "false"]
        self.train_proportion = train_proportion
        self.val_proportion = val_proportion
        self.fws_random_state = fws_random_state
        self.train_specs = train_specs
        self.storage_path = storage_path

        self.model = self.get_model()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_specs["lr"],
            weight_decay=self.train_specs["weight_decay"],
        )

        # logging stuff
        self.train_loss = []
        self.val_loss = []

        self.train_accuracy = []
        self.val_accuracy = []

        self.train_f1 = []
        self.val_f1 = []

        self.train_mcc = []
        self.val_mcc = []

        self.train_roc_auc = []
        self.val_roc_auc = []

        # Save model specs
        model_specs_df = pd.DataFrame.from_dict(model_specs, orient="index")
        model_specs_df.to_csv(os.path.join(storage_path, "model_specs.csv"))

    # def load_data(self, train_val_together=False):
    #     gene_families_file = "Data/Processed/gene_families.csv"  # FIXED
    #     data_path = "Data/Processed"  # FIXED
    #     train_proportion = self.train_proportion
    #     validation_proportion = self.val_proportion

    #     data_handler = DataHandler(
    #         self.DNA_specs,
    #         gene_families_file,
    #         data_path,
    #         train_proportion,
    #         validation_proportion,
    #         random_state=self.fws_random_state,
    #     )
    #     mRNA_train, mRNA_validation, mRNA_test = data_handler.get_data(
    #         fdr=0.01
    #     )  # FIXED and IMPORTANT
    #     train_loader = get_dataloader(
    #         mRNA_train,
    #         data_handler.TSS_sequences,
    #         data_handler.TTS_sequences,
    #         None if self.DNA_specs[4] == "false" else data_handler.intron_sequences,
    #         batch_size=self.train_specs["batch_size"],
    #         shuffle=True,
    #     )

    #     validation_loader = get_dataloader(
    #         mRNA_validation,
    #         data_handler.TSS_sequences,
    #         data_handler.TTS_sequences,
    #         None if self.DNA_specs[4] == "false" else data_handler.intron_sequences,
    #         batch_size=self.train_specs["batch_size"],
    #         shuffle=True,
    #     )

    #     test_loader = get_dataloader(
    #         mRNA_test,
    #         data_handler.TSS_sequences,
    #         data_handler.TTS_sequences,
    #         None if self.DNA_specs[4] == "false" else data_handler.intron_sequences,
    #         batch_size=self.train_specs["batch_size"],
    #         shuffle=False,
    #     )

    #     if train_val_together:
    #         train_loader = get_dataloader(
    #             pd.concat([mRNA_train, mRNA_validation]),
    #             data_handler.TSS_sequences,
    #             data_handler.TTS_sequences,
    #             None if self.DNA_specs[4] == "false" else data_handler.intron_sequences,
    #             batch_size=self.train_specs["batch_size"],
    #             shuffle=True,
    #         )

    #         return train_loader, test_loader

    #     return train_loader, validation_loader, test_loader

    # def handle_batch(self, batch):
    #     TTS = batch["TTS"]
    #     TSS = batch["TSS"]
    #     padding_20bp = np.zeros((TSS.shape[0], 20, 4))
    #     if self.DNA_specs[4] == "true":
    #         intron = batch["introns"]
    #         DNA = np.concatenate((TSS, padding_20bp, TTS, padding_20bp, intron), axis=1)
    #         DNA = torch.tensor(DNA).float().to(self.device).transpose(1, 2)
    #     else:
    #         DNA = np.concatenate((TSS, padding_20bp, TTS), axis=1)
    #         DNA = torch.tensor(DNA).float().to(self.device).transpose(1, 2)

    #     labels = torch.tensor(batch["DE"]).float().to(self.device)

    #     return DNA, labels

    def get_model(self):
        """
        Get the model
        """
        print(
            "Getting the model with required specifications:"
            f"\nDNA_specs: {self.DNA_specs}, \nmodel_specs: {self.model_specs}"
        )

        model = myCNN(
            sequence_length=(
                np.sum(self.DNA_specs[:4]) + 20
                if self.DNA_specs[4] == "false"
                else np.sum(self.DNA_specs[:4]) + 40 + 2000
            ),
            n_labels=5,  # FIXED
            n_ressidual_blocks=self.model_specs["n_ressidual_blocks"],
            in_channels=4,  # FIXED
            out_channels=self.model_specs["out_channels"],
            kernel_size=self.model_specs["kernel_size"],
            max_pooling_kernel_size=self.model_specs["max_pooling_kernel_size"],
            dropout_rate=self.model_specs["dropout_rate"],
            ffn_size_1=self.model_specs["ffn_size_1"],
            ffn_size_2=self.model_specs["ffn_size_2"],
        )

        return model.to(self.device)

    def train_model(self):
        """
        Train the model. Returns best performance metrics in the validation set.
        Stores best performing model in terms of loss in the storage path.
        """

        # load data
        train_loader, validation_loader, _ = self.load_data()

        scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=5, factor=0.5, verbose=True
        )
        min_loss = np.inf
        for epoch in range(self.train_specs["n_epochs"]):
            output_list_train = []
            output_list_validation = []
            label_list_train = []
            label_list_validation = []
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                DNA, labels = self.handle_batch(batch)
                self.optimizer.zero_grad()

                outputs = self.model(DNA)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                output_list_train.append(outputs.cpu().detach().numpy().flatten())
                label_list_train.append(labels.cpu().detach().numpy().flatten())

            output_list_train = np.concatenate(output_list_train)
            label_list_train = np.concatenate(label_list_train)

            # Use sigmoid elementwise to get the labels
            output_list_train = 1 / (1 + np.exp(-output_list_train))

            accuracy = accuracy_score(label_list_train, output_list_train > 0.5)
            f1 = f1_score(label_list_train, output_list_train > 0.5)
            mcc = matthews_corrcoef(label_list_train, output_list_train > 0.5)
            auc = roc_auc_score(label_list_train, output_list_train)

            print(
                f"Epoch {epoch}, \nTRAIN: average loss: {running_loss / len(train_loader):.2f}, accuracy: {accuracy:.2f}, f1: {f1:.2f}, mcc: {mcc:.2f}, auc: {auc:.2f}"
            )

            self.train_loss.append(running_loss / len(train_loader))
            self.train_accuracy.append(accuracy)
            self.train_f1.append(f1)
            self.train_mcc.append(mcc)
            self.train_roc_auc.append(auc)

            self.model.eval()
            running_loss = 0.0
            for batch in validation_loader:
                DNA, labels = self.handle_batch(batch)
                outputs = self.model(DNA)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                output_list_validation.append(outputs.cpu().detach().numpy().flatten())
                label_list_validation.append(labels.cpu().detach().numpy().flatten())

            output_list_validation = np.concatenate(output_list_validation)
            label_list_validation = np.concatenate(label_list_validation)

            output_list_validation = 1 / (1 + np.exp(-output_list_validation))

            accuracy_val = accuracy_score(
                label_list_validation, output_list_validation > 0.5
            )
            f1_val = f1_score(label_list_validation, output_list_validation > 0.5)
            mcc_val = matthews_corrcoef(
                label_list_validation, output_list_validation > 0.5
            )
            auc_val = roc_auc_score(label_list_validation, output_list_validation)

            scheduler.step(running_loss / len(validation_loader))

            print(
                f"VALIDATION: average loss: {running_loss / len(validation_loader):.2f}, accuracy: {accuracy_val:.2f}, f1: {f1_val:.2f}, mcc: {mcc_val:.2f}, auc: {auc_val:.2f}"
            )

            self.val_loss.append(running_loss / len(validation_loader))
            self.val_accuracy.append(accuracy_val)
            self.val_f1.append(f1_val)
            self.val_mcc.append(mcc_val)
            self.val_roc_auc.append(auc_val)

            if running_loss / len(validation_loader) < min_loss:
                epochs_no_improve = 0
                min_loss = running_loss / len(validation_loader)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.storage_path, "best_model.pt"),
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 15:
                    print("Early stopping")
                    break

        # Get the index of best validation loss, report the metrics for that epoch
        best_epoch = np.argmin(self.val_loss)
        assert np.min(self.val_loss) == min_loss
        best_val_loss = self.val_loss[best_epoch]
        best_val_accuracy = self.val_accuracy[best_epoch]
        best_val_f1 = self.val_f1[best_epoch]
        best_val_mcc = self.val_mcc[best_epoch]
        best_val_auc = self.val_roc_auc[best_epoch]

        print(
            f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.2f}, best val accuracy: {best_val_accuracy:.2f}, best val f1: {best_val_f1:.2f}, best val mcc: {best_val_mcc:.2f}, best val auc: {best_val_auc:.2f}"
        )
        print(
            f"Training metrics: accuracy: {self.train_accuracy[best_epoch]:.2f}, f1: {self.train_f1[best_epoch]:.2f}, mcc: {self.train_mcc[best_epoch]:.2f}, auc: {self.train_roc_auc[best_epoch]:.2f}"
        )

        return {
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "best_val_f1": best_val_f1,
            "best_val_mcc": best_val_mcc,
            "best_val_auc": best_val_auc,
        }

        # Once training is finished

    def train_and_test(self, val_loss):
        """
        Given best hyperparameters and known validation best loss. Train from scratch on
        train + validation data and test on test data.
        """
        print("Training the model on train + validation data")
        print("Assuming it is using the best hyperparameters")
        # load data
        train_loader, test_loader = self.load_data(train_val_together=True)

        self.model = self.get_model()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.train_specs["lr"],
            weight_decay=self.train_specs["weight_decay"],
        )
        training_loss = np.inf
        while training_loss > val_loss:
            training_losses = []
            for batch in train_loader:
                DNA, labels = self.handle_batch(batch)
                self.optimizer.zero_grad()

                outputs = self.model(DNA)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                training_losses.append(loss.item())

            training_loss = np.mean(training_losses)
            print(f"Training loss: {training_loss}")

        # Test the model
        print("Testing the model")
        self.model.eval()
        output_list_test = []
        label_list_test = []
        running_loss = 0.0
        for batch in test_loader:
            DNA, labels = self.handle_batch(batch)
            outputs = self.model(DNA)
            loss = self.loss_fn(outputs, labels)
            running_loss += loss.item()

            output_list_test.append(outputs.cpu().detach().numpy().flatten())
            label_list_test.append(labels.cpu().detach().numpy().flatten())

        output_list_test = np.concatenate(output_list_test)
        label_list_test = np.concatenate(label_list_test)

        output_list_test = 1 / (1 + np.exp(-output_list_test))

        accuracy_test = accuracy_score(label_list_test, output_list_test > 0.5)
        f1_test = f1_score(label_list_test, output_list_test > 0.5)
        mcc_test = matthews_corrcoef(label_list_test, output_list_test > 0.5)
        auc_test = roc_auc_score(label_list_test, output_list_test)

        # Make a classification report
        print(classification_report(label_list_test, output_list_test > 0.5))
        # Make a confusion matrix
        print(confusion_matrix(label_list_test, output_list_test > 0.5))
        # Make a calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            label_list_test, output_list_test, n_bins=10
        )

        plt.plot(mean_predicted_value, fraction_of_positives, marker="o")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.xlabel("Mean predicted value")
        plt.ylabel("Fraction of positives")
        plt.savefig(os.path.join(self.storage_path, "calibration_curve.png"))

        print(
            f"TEST: average loss: {running_loss / len(test_loader):.2f}, accuracy: {accuracy_test:.2f}, f1: {f1_test:.2f}, mcc: {mcc_test:.2f}, auc: {auc_test:.2f}"
        )


if __name__ == "__main__":
    model_specs = {
        "n_ressidual_blocks": 4,
        "out_channels": 122,
        "kernel_size": 6,
        "max_pooling_kernel_size": 4,
        "dropout_rate": 0.5,
        "ffn_size_1": 128,
        "ffn_size_2": 64,
    }

    DNA_specs = [1500, 500, 500, 1500, "false"]
    train_proportion = 0.85
    val_proportion = 0.15
    train_specs = {
        "lr": 0.000001,
        "weight_decay": 0.006,
        "n_epochs": 100,  # FIXED
        "batch_size": 64,  # FIXED
    }
    storage_path = "Results/Models/CNN/Model1"
    os.makedirs(storage_path, exist_ok=True)
    trainer = Trainer(
        model_specs,
        DNA_specs,
        train_proportion,
        val_proportion,
        train_specs,
        storage_path,
    )

    #best_val_metrics = trainer.train_model()

    # Test

    trainer.train_and_test(0.57)#best_val_metrics["best_val_loss"])
