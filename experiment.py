#!/usr/bin/env python
import argparse
import os
import glob
import csv
import time
from torch.utils.data import DataLoader
from dataloaders import *
from models import *
import pandas as pd
import schedulefree
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class Experiment:
    def __init__(self,
                 task: str, 
                 model_type: str, 
                 dataset_name: str, 
                 device, 
                 input_dim: int, 
                 n_samples: int, 
                 noise_prob: float, 
                 length_range: tuple[int,int], 
                 value_range, #1 case where its not tuple[int,int] so no type hinting here for that... sorry.
                 target_range: tuple[int,int], 
                 weight_range: tuple[int,int], 
                 adversarial_range: tuple[int,int],
                 top_cat: str = 'eighth_experiment',
                 experiment_type: str = 'train', 
                 batch_size: int = 64, 
                 seed: int = 0, 
                 num_epochs: int = 50, 
                 lr: float = 0.001, 
                 d_model: int = 32, 
                 n_heads: int = 2, 
                 num_layers: int = 1, 
                 dropout: float = 0.0): 
        dict_datasets = {
            'SubsetSumDecisionDataset': {'class': SubsetSumDecisionDataset, 'classification': True, 'pool': True}, 
            'MaxSubsetSumDataset': {'class': MaxSubsetSumDataset, 'classification': False, 'pool': True}, 
            'KnapsackDataset': {'class': KnapsackDataset, 'classification': False, 'pool': True},
            'FractionalKnapsackDataset': {'class': FractionalKnapsackDataset, 'classification': False, 'pool': True},
            'MinCoinChangeDataset': {'class': MinCoinChangeDataset, 'classification': False, 'pool': True},
            'QuickselectDataset': {'class': QuickselectDataset, 'classification': True, 'pool': False},
            'BalancedPartitionDataset': {'class': BalancedPartitionDataset, 'classification': False, 'pool': True},
            'BinPackingDataset': {'class': BinPackingDataset, 'classification': False, 'pool': True},
            'ConvexHullDataset': {'class': ConvexHullDataset, 'classification': True, 'pool': False}, 
            'ThreeSumDecisionDataset': {'class': ThreeSumDecisionDataset, 'classification': True, 'pool': True},
            'FloydWarshallDataset': {'class': FloydWarshallDataset, 'classification': False, 'pool': False},
            'SCCDataset': {'class': SCCDataset, 'classification': True, 'pool': False},
            'LISDataset': {'class': LISDataset, 'classification': False, 'pool': True}
        }
        # -- assertions -- 
        assert task in ['evaluate', 'train']
        assert model_type in ['tropical', 'vanilla', 'adaptive']
        assert dataset_name in list(dict_datasets.keys())

        # -- essentials --
        self.task = task
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        self.dict_dataset = dict_datasets[self.dataset_name]
        
        # -- data --
        self.device = device
        self.input_dim = input_dim
        self.length_range = length_range
        self.n_samples = n_samples
        self.noise_prob = noise_prob
        self.value_range = value_range
        self.target_range = target_range
        self.weight_range = weight_range
        self.adversarial_range = adversarial_range  
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.lr = lr
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # -- tracking --
        self.top_cat = top_cat
        self.init_time = self._time_string()
        self.base_pattern = f"{self.dataset_name}_{self.model_type}"
        if task == 'evaluate':
            self.full_file_name =  f"{self.base_pattern}_{self.length_range}_{self.noise_prob}_{self.value_range}_{self.init_time}"
        else:  
            self.full_file_name =  f"{self.base_pattern}_{self.lr}_{self.num_epochs}_{self.init_time}"
        print(self.full_file_name)

    def run(self):
        self.set_dataloader() 
        self.set_model() 
        if self.task == 'evaluate':
            pattern = os.path.join(self.top_cat, "models", f"{self.base_pattern}*.pth")  
            matching_files = glob.glob(pattern)
            state_dict = torch.load(matching_files[0], map_location=self.device)
            self.model_being_evaluated = matching_files[0].replace('.pth','').replace(os.path.join(self.top_cat, "models"),'').replace('/','') # change back to "/"
            print(self.model_being_evaluated)
            self.model.load_state_dict(state_dict)
            self.evaluate_model()
        else:
            self.train_model()
        
    def set_model(self):
        print(f'...setting model object...{self._time_string()}')
        self.model = SimpleTransformerModel(input_dim=self.input_dim,
                                            d_model = self.d_model,
                                            n_heads = self.n_heads,
                                            num_layers = self.num_layers,
                                            dropout = self.dropout,
                                            tropical = (self.model_type == 'tropical'),
                                            tropical_attention_cls = TropicalAttention(self.d_model, self.n_heads, self.device) if self.model_type == 'tropical' else None,
                                            classification=self.dict_dataset['classification'],
                                            pool=self.dict_dataset['pool'],
                                            aggregator='softmax' if self.model_type == 'vanilla' else 'adaptive').to(self.device)
    
    def _save_model(self):
        model_folder = os.path.join(self.top_cat, 'models')
        os.makedirs(model_folder, exist_ok=True)
        model_path = f"{model_folder}/{self.full_file_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"...model saved to {model_path}...{self._time_string()}")
    
    def set_dataloader(self):
        print(f'...setting dataloader...{self._time_string()}')
        
        self.dataset = self.dict_dataset['class'](n_samples = self.n_samples,
                                                                length_range = self.length_range,
                                                                value_range = self.value_range,
                                                                target_range = self.target_range,
                                                                weight_range = self.weight_range,
                                                                adversarial_range = self.adversarial_range,
                                                                noise_prob = self.noise_prob,
                                                                classification =self.dict_dataset['classification'], 
                                                                seed = self.seed,)
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=(self.task == 'train'))
        val_length_range = (self.length_range[0]*2,self.length_range[1]*2)
        if self.dataset_name == "SCCDataset":
            val_value_range = 0.3
        else:
            val_value_range = (self.value_range[0]*2 if (self.value_range[0] not in [0,1]) else self.value_range[0],self.value_range[1]*2)
        self.val_dataset = self.dict_dataset['class'](n_samples = int(self.n_samples/10),
                                                                length_range = val_length_range,
                                                                value_range = val_value_range,
                                                                target_range = self.target_range,
                                                                weight_range = self.weight_range,
                                                                adversarial_range = self.adversarial_range,
                                                                noise_prob = self.noise_prob,
                                                                classification =self.dict_dataset['classification'], 
                                                                seed = self.seed,)
        print(val_length_range)
        print(val_value_range)
        self.val_dataloader = DataLoader(self.val_dataset, 
                                     batch_size=self.batch_size,
                                     shuffle=(self.task == 'train'))

    def _eval_one_epoch(self, type="test"):
        self.model.eval()
        losses, total_loss = [], 0.0
        all_preds, all_targets, all_masks = [], [], []   # masks used only for Quickselect
        use_log_scale = False
        dl_to_use = self.dataloader if type == "test" else self.val_dataloader

        with torch.no_grad():
            for x, y in dl_to_use:                        # x: (B, n, d), y: (B, 1) or (B, …)
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)                     # shape depends on model

                # ---------- classification ----------
                if self.model.classification:
                    if self.dataset_name in ["QuickselectDataset", "SCCDataset", "ConvexHullDataset"]:
                        # ----- CLRS pointer evaluation -----
                        logits = pred.squeeze(-1)                       # (B, n)
                        batch_preds = (torch.sigmoid(logits) > 0.5).long()  # (B, n)
                        batch_targets = y.squeeze(-1).long()               # (B, n)
                        node_mask = (x.abs().sum(dim=-1) != 0)            # (B, n) – padded nodes

                        all_preds.append(batch_preds.cpu())
                        all_targets.append(batch_targets.cpu())
                        all_masks.append(node_mask.cpu())

                        batch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits,
                            y.squeeze(-1).float()
                        )

                    else:
                        # your existing non-Quickselect classification logic…
                        if self.model.pool:
                            batch_preds = (torch.sigmoid(pred.squeeze(-1)) > 0.5).long()
                        else:
                            batch_preds = torch.argmax(pred, dim=-1)

                        all_preds.append(batch_preds.cpu())
                        all_targets.append(y.cpu())

                        batch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            pred.squeeze(-1), y.squeeze(-1).float()
                        )

                # ---------- regression ----------
                else:
                    if use_log_scale:
                        batch_loss = torch.nn.functional.mse_loss(
                            torch.log1p(torch.nn.functional.relu(pred)),
                            torch.log1p(torch.nn.functional.relu(y)),
                        ).sqrt()
                    else:
                        batch_loss = torch.nn.functional.mse_loss(pred, y)

                losses.append(batch_loss.item())
                total_loss += batch_loss.item() * x.size(0)

        # -------- aggregate metrics --------
        avg_loss = total_loss / len(dl_to_use.dataset)
        std_loss = float(np.std(losses, ddof=0))

        if self.model.classification:
            if self.dataset_name in ["QuickselectDataset", "SCCDataset", "ConvexHullDataset"]:
                preds_flat   = torch.cat(all_preds, 0).view(-1).numpy()              # (total_positions,)
                targets_flat = torch.cat(all_targets, 0).view(-1).numpy()            # (total_positions,)
                mask_flat    = torch.cat(all_masks,  0).view(-1).numpy().astype(bool) # (total_positions,)

                micro_f1 = f1_score(
                    targets_flat[mask_flat],
                    preds_flat[mask_flat],
                    average="binary",    # binary ≡ micro for a single positive class
                    zero_division=0,
                )
            else:
                preds_cat   = torch.cat(all_preds,   0).view(-1).numpy()
                targets_cat = torch.cat(all_targets, 0).view(-1).numpy()
                micro_f1 = f1_score(targets_cat, preds_cat, average="micro")

            return avg_loss, std_loss, micro_f1

        # regression – no F1
        return avg_loss, std_loss, None


    
    def evaluate_model(self):
        print(f'...evaluating model...{self._time_string()}')
        loss, std, micro_f1_score = self._eval_one_epoch()
        if os.path.exists(f'{self.top_cat}/evaluate/{self.experiment_type}_test/evaluate_{self.model_being_evaluated}.csv') == False:
            self._write_to_csv('w', 'evaluate', ['seed','length_range', 'noise_prob', 'value_range', 'loss', 'std', 'micro_f1_score'])
        self._write_to_csv('a', 'evaluate', [self.seed, self.length_range, self.noise_prob, self.value_range, loss, std, micro_f1_score])
    
    def _time_string(self):
        return time.strftime("%Y%m%d_%H%M%S",time.localtime())
    
    def _write_to_csv(self, mode, task, list_data):
        task_folder = os.path.join(self.top_cat, task)
        os.makedirs(task_folder, exist_ok=True)
        if task == 'evaluate':
            experiment_folder = os.path.join(task_folder, f'{self.experiment_type}_test')
            os.makedirs(experiment_folder, exist_ok=True)
            csv_path = os.path.join(experiment_folder, f"{task}_{self.model_being_evaluated}.csv")
            with open(csv_path, mode=mode, newline="") as file:
                writer = csv.writer(file)
                writer.writerow(list_data)
        else:
            csv_path = os.path.join(task_folder, f"{task}_{self.full_file_name}.csv")
            with open(csv_path, mode=mode, newline="") as file:
                writer = csv.writer(file)
                writer.writerow(list_data)

    def _plot_training_run(self):
        plots_folder = os.path.join(self.top_cat, 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        output_file = os.path.join(plots_folder, f'plots_{self.full_file_name}.png')
        f_name = f"{os.path.join(self.top_cat, 'train')}/train_{self.full_file_name}.csv"
        data = pd.read_csv(f_name)
        data.sort_values(by=['epoch', 'batch'], inplace=True)
        data['global_step'] = data.index
        plt.figure(figsize=(10,6))
        plt.plot(data['global_step'], data['loss'], linestyle='-', label='loss')

        epochs = data['epoch'].unique()
        y_max = data['loss'].max()
        for epoch in epochs:
            first_idx = data[data['epoch'] == epoch]['global_step'].iloc[0]
            plt.axvline(x=first_idx, color='gray', linestyle='--', alpha=0.5)
            plt.text(first_idx, y_max, f'Epoch {epoch}', rotation=90, verticalalignment='bottom', fontsize=8, color='gray')
        loss_type = "BCE" if self.dict_dataset['classification'] else "MSE"
        plt.xlabel('Global Step (cumulative batch index)')
        plt.ylabel(f'{loss_type} Loss')
        plt.title(f'{loss_type} Loss Progression Across Epochs')
        plt.legend()
        plt.grid(True)

        plt.savefig(output_file)
        plt.close()

        print(f'...plot saved to {output_file}...')

    def _train_one_epoch(self, epoch):
        self.model.train()
        ###self.optimizer.train()
        total_loss_val = 0
        use_log_scale = False
        #print(self.model.classification, self.model.output_linear.out_features)
        for i, (x, y) in enumerate(self.dataloader, start=1):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            #print(pred.size(), y.size(), self.model.pool)
            if self.model.classification:# Check if we are in classification mode.
                #if self.model.pool: # Binary classification (n_classes == 1)
                    #loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), y.float()) # Squeeze prediction to match target shape: [batch, seq_len]
                #else:
                if self.dataset_name == "QuickselectDataset":
                    loss = F.cross_entropy(pred.squeeze(-1), y.squeeze(-1)) 
                else:
                    loss = F.binary_cross_entropy_with_logits(pred.squeeze(-1), y.squeeze(-1).float()) 
            else:
                # Use log-scale for regression if enabled.
                if use_log_scale:
                    loss = F.mse_loss(torch.log1p(F.relu(pred)), torch.log1p(F.relu(y))).sqrt()
                else:
                    loss = F.mse_loss(pred, y)
            l1_reg = 0.0
            for param in self.model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss = loss + 0.000 * l1_reg
            total_loss = loss
            total_loss.backward()
            self.optimizer.step()
            total_loss_val += total_loss.item() * x.size(0)
            if i % 10 == 0:
                #self._write_to_csv('a', 'train', [epoch, i, total_loss_val/(min(i*self.batch_size, self.n_samples)), time.time()])
                self._write_to_csv('a', 'train', [epoch, i, total_loss.item(), time.time()])
        return total_loss_val / self.n_samples

    def train_model(self):
        print(f'...training model...{self._time_string()}')
        ###self.optimizer = schedulefree.RAdamScheduleFree(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self._write_to_csv('w', 'train', ['epoch', 'batch', 'loss', 'time'])
        #self._write_to_csv('w', 'validation', ['epoch', 'val_loss', 'val_f1', 'best_up_to_now', 'time'])
        #loss_measure = True
        ##best_metric = 100000.0
        for epoch in range(self.num_epochs):
            ##best_this_round = 'No'
            self._train_one_epoch(epoch)
            ##val_loss, _, val_f1 = self._eval_one_epoch(type='validation')
            ##if (val_loss < best_metric and epoch > 25) or epoch == 0:
                ##best_this_round = 'Yes'
                ##self._save_model()
                ##best_metric = val_loss
            ##self._write_to_csv('a', 'validation', [epoch, val_loss, val_f1, best_this_round, time.time()])
        self._plot_training_run()
        self._save_model()

def convert_value(value):
    try:
        # Attempt to evaluate the value to its corresponding Python type
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If it fails, return the original value (likely a plain string)
        return value
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", type=str, default='jobs_to_do_train', help="job file to read")
    parser.add_argument("--job_id", type=int, default=23, help="job_id index to select model, dataset, and layer type")
    args = parser.parse_args()

    df = pd.read_csv(f'{args.job_file}.csv')
    #for i in range(216): raw_params = df.iloc[i].to_dict()
    raw_params = df.iloc[args.job_id].to_dict()
    print(raw_params)
    config_params = {key: convert_value(val) for key, val in raw_params.items()}
    experiment = Experiment(device = 'cuda' if torch.cuda.is_available() else 'cpu', 
                            **config_params)
    experiment.run()


