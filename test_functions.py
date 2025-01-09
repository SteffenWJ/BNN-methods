import os
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import bayesian_torch.layers as bnn_layers # used in bayes by backprop
import pandas as pd

from laplace import Laplace

from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

from helper_functions import show_image_confidence


def metric_NLL(y_true, y_pred_probs):
    epsilon = 1e-12
    nll = -np.log(y_pred_probs[range(len(y_true)), y_true] + epsilon)
    return np.mean(nll)

def metric_brier_score(y_true, y_pred_probs):
    true_probs = np.zeros_like(y_pred_probs)
    true_probs[range(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_pred_probs - true_probs) ** 2, axis=1))

def metric_ECE(y_true, y_pred_probs, num_bins=10):
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(predictions[in_bin] == y_true[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece


def extract_variances(model):
    variances = {}
    found = False

    for name, layer in model.named_modules():
        if isinstance(layer, bnn_layers.Conv2dReparameterization):
            found = True

            # Extract rho_kernel and compute sigma for kernel
            rho_kernel = layer.rho_kernel.data
            sigma_kernel = F.softplus(rho_kernel)
            variances[f'{name}.sigma_kernel'] = sigma_kernel.cpu().numpy()

            # Extract rho_bias and compute sigma for bias
            if hasattr(layer, 'rho_bias') and layer.rho_bias is not None:
                rho_bias = layer.rho_bias.data
                sigma_bias = F.softplus(rho_bias)
                variances[f'{name}.sigma_bias'] = sigma_bias.cpu().numpy()

        elif isinstance(layer, bnn_layers.LinearReparameterization):
            found = True

            rho_weight = layer.rho_weight.data
            sigma_weight = F.softplus(rho_weight)
            variances[f'{name}.sigma_weight'] = sigma_weight.cpu().numpy()

            if hasattr(layer, 'rho_bias') and layer.rho_bias is not None:
                rho_bias = layer.rho_bias.data
                sigma_bias = F.softplus(rho_bias)
                variances[f'{name}.sigma_bias'] = sigma_bias.cpu().numpy()

    if not found:
        raise ValueError(
            "No Conv2dReparameterization or LinearReparameterization layers found in the model."
        )

    return variances

def variances_to_dataframe(variances: Dict[str, Any]) -> pd.DataFrame:
    data = []
    for key, values in variances.items():
        layer, param = key.rsplit('.', 1) 
        for sigma in values.flatten():      
            data.append({'Layer': layer, 'Parameter': param, 'Sigma': sigma})

    df = pd.DataFrame(data)
    return df

def plot_mean_and_std(df: pd.DataFrame, save: bool = False, save_path: str = "mean_std_plot.png", figsize: tuple = (12, 6)):
    sns.set_theme(style="whitegrid")
    summary = df.groupby(['Layer', 'Parameter'])['Sigma'].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=summary, x='Layer', y='mean', hue='Parameter', capsize=.2, palette='viridis')
    layer_order = summary['Layer'].unique()
    parameter_order = summary['Parameter'].unique()
    #n_layers = len(layer_order)
    n_parameters = len(parameter_order)
    width = 0.8
    for i, layer in enumerate(layer_order):
        for j, param in enumerate(parameter_order):
            subset = summary[(summary['Layer'] == layer) & (summary['Parameter'] == param)]
            if not subset.empty:
                mean = subset['mean'].values[0]
                std = subset['std'].values[0]
                offset = (j - n_parameters / 2) * width / n_parameters + width / (2 * n_parameters)
                x = i + offset
                ax.errorbar(x=x, y=mean, yerr=std, fmt='none', c='black', capsize=5)

    plt.title('Mean and Standard Deviation of Variances per Layer and Parameter')
    plt.xlabel('Layer')
    plt.ylabel('Mean Sigma')
    plt.xticks(rotation=45)
    plt.legend(title='Parameter')

    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Mean and Standard Deviation plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_boxplots(df: pd.DataFrame, save: bool = False, save_path: str = "boxplots.png", figsize: tuple = (14, 7)):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x='Layer', y='Sigma', hue='Parameter', palette='Set2')

    # Customize the plot
    plt.title('Boxplots of Variances per Layer and Parameter')
    plt.xlabel('Layer')
    plt.ylabel('Sigma')
    plt.xticks(rotation=45)
    plt.legend(title='Parameter')

    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Boxplots saved to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_variances(df: pd.DataFrame, save: bool = False, save_dir: str = "plots"):
    if save:
        os.makedirs(save_dir, exist_ok=True)
        mean_std_path = os.path.join(save_dir, "mean_std_plot.png")
        boxplot_path = os.path.join(save_dir, "boxplots.png")
    else:
        mean_std_path = None
        boxplot_path = None

    # Plot Mean and Standard Deviation
    plot_mean_and_std(df, save=save, save_path=mean_std_path)

    # Plot Boxplots
    plot_boxplots(df, save=save, save_path=boxplot_path)
    
    
def plot_output_variances(proberbilty_output, save=False, save_path=None):
    confidence_mean = proberbilty_output.mean(axis=0)
    confidence_std = proberbilty_output.std(axis=0)
    #confidence_max = confidence_mean + confidence_std
    #confidence_min = confidence_mean - confidence_std
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(confidence_mean)), confidence_mean, yerr=confidence_std, capsize=3, color='skyblue', label='Mean Prediction')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.7)  # Add a line at y=0 for reference
    plt.xlabel('Index')
    plt.ylabel('Mean Prediction')
    plt.title('Predictions with Uncertainty')

    mean_patch = mpatches.Patch(color='skyblue', label='Mean Prediction')
    std_line = mlines.Line2D([], [], color='black', marker='_', linestyle='None', markersize=10, label='Standard Deviation')

    plt.legend(handles=[mean_patch, std_line])

    plt.show()
    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Output variances plot saved to {save_path}")
    plt.close() #Should clean memory
    


class model_evaluater:
    def __init__(self, model, data_loader, image_paramters=None, ensamble=False, dropout=False, criterion=None, device='cpu'):
        self.model = model
        self.device = device
        if ensamble:
            for temp in self.model:
                temp.train() if dropout else temp.eval()
                temp.to(self.device)
        else:
            self.model.train() if dropout else self.model.eval()
            model.to(self.device)

        self.ensamble = ensamble
        self.dropout = dropout

        self.data_loader = data_loader
        self.image_paramters = image_paramters

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        self.metrics = None
        
        self.laplace = None

    def evaluate(self, samples_per_model=1, ece_bins=10):
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_targets = []
        aleatoric_variances = []
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_aleatoric_variances = []
                if self.laplace is not None:
                    model_outputs = self.laplace.predictive_samples(inputs, n_samples=samples_per_model)
                    list_of_sampled_logits = model_outputs
                else:    
                    list_of_sampled_logits = []
                    for _ in range(samples_per_model):
                        if not self.ensamble:
                            model_outputs = self.model(inputs)
                        else:
                            model_outputs = []
                            for tmp in self.model:
                                model_outputs.append(tmp(inputs))
                            model_outputs = torch.mean(torch.stack(model_outputs), dim=0)
                        if isinstance(model_outputs, (tuple, list)):
                            # In case of Aletoric Uncertainty
                            mean = model_outputs[0]
                            log_var = model_outputs[1]
                            sigma = torch.exp(0.5 * log_var)
                            epsilon = torch.randn_like(mean)
                            sampled_logits = mean + epsilon * sigma
                            aleatoric_variance = sigma.pow(2)
                            batch_aleatoric_variances.append(aleatoric_variance.cpu())
                        else:
                            sampled_logits = model_outputs
                        list_of_sampled_logits.append(sampled_logits)
                    list_of_sampled_logits = torch.stack(list_of_sampled_logits, dim=0)
                logits = torch.mean(list_of_sampled_logits, dim=0)
                if batch_aleatoric_variances:
                    avg_aleatoric_variance = torch.stack(batch_aleatoric_variances, dim=0).mean(dim=0)
                    aleatoric_variances.append(avg_aleatoric_variance)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                all_probs.append(probabilities.cpu())
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        if aleatoric_variances:
            aleatoric_variances = torch.cat(aleatoric_variances)
            mean_aleatoric_variance = aleatoric_variances.mean().item()
        else:
            aleatoric_variances = None
            mean_aleatoric_variance = None
        nll_score = metric_NLL(all_targets, all_probs)
        brier_score = metric_brier_score(all_targets, all_probs)
        ece_score = metric_ECE(all_targets, all_probs, ece_bins)
        average_loss = total_loss / len(self.data_loader)

        self.metrics = {
            'Average Loss': average_loss,
            'NLL': nll_score,
            'Brier Score': brier_score,
            'ECE': ece_score,
            'All_aleatoric_variances': aleatoric_variances,
            'Mean Aleatoric Variance': mean_aleatoric_variance,
            'All_probs': all_probs,
            'All_targets': all_targets
        }


    def get_metrics(self):
        if self.metrics is None:
            raise ValueError('Metrics are not calculated yet. Please run evaluate() method first.')
        else:
            return self.metrics

    def print_metrics(self, long=False):
        if self.metrics is None:
            raise ValueError('Metrics are not calculated yet. Please run evaluate() method first.')
        else:
            if long:
                print(f'Average Loss: {self.metrics["Average Loss"]}')
                print(f'NLL: {self.metrics["NLL"]}')
                print(f'Brier Score: {self.metrics["Brier Score"]}')
                print(f'ECE: {self.metrics["ECE"]}')
                if 'Mean Aleatoric Variance' in self.metrics and self.metrics['Mean Aleatoric Variance'] is not None:
                    print(f'Mean Aleatoric Variance: {self.metrics["Mean Aleatoric Variance"]}')
            else:
                print(f'Average Loss: {self.metrics["Average Loss"]:.4f}')
                print(f'NLL: {self.metrics["NLL"]:.4f}')
                print(f'Brier Score: {self.metrics["Brier Score"]:.4f}')
                print(f'ECE: {self.metrics["ECE"]:.4f}')
                if 'Mean Aleatoric Variance' in self.metrics and self.metrics['Mean Aleatoric Variance'] is not None:
                    print(f'Mean Aleatoric Variance: {self.metrics["Mean Aleatoric Variance"]:.4f}')
        
    def plot_combined_ece_analysis(self, probs, labels, num_bins=10, show_plot = False, save=False, save_path=None, epoch=None):
        if save and save_path is None:
            raise ValueError("When save=True, save_path must be specified.")
        
        if save:
            os.makedirs(save_path, exist_ok=True)
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        samples_in_bin = []
        
        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            if bin_upper == 1.0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            num_in_bin = np.sum(in_bin)
            samples_in_bin.append(num_in_bin)
            if num_in_bin > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
            else:
                bin_accuracy = 0
                bin_confidence = 0
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            
            if epoch is not None:
                plot_names = {
                    'Reliability_Diagram': 'Reliability_Diagram_epoch_{}.png'.format(epoch),
                    'Confidence_Histogram': 'Confidence_Histogram_epoch_{}.png'.format(epoch),
                    'Confidence_vs_Accuracy': 'Confidence_vs_Accuracy_epoch_{}.png'.format(epoch),
                    'Samples_per_Bin': 'Samples_per_Bin_epoch_{}.png'.format(epoch)
                }
            else:
                plot_names = {
                    'Reliability_Diagram': 'Reliability_Diagram.png',
                    'Confidence_Histogram': 'Confidence_Histogram.png',
                    'Confidence_vs_Accuracy': 'Confidence_vs_Accuracy.png',
                    'Samples_per_Bin': 'Samples_per_Bin.png'
                }
            
        plots = {
            'Reliability_Diagram': {
                'data': (bin_confidences, bin_accuracies),
                'plot_func': lambda axs: (
                    axs.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration'),
                    axs.plot(bin_confidences, bin_accuracies, marker='o', label='Model Calibration'),
                    axs.fill_between(bin_confidences, bin_accuracies, bin_confidences, color='blue', alpha=0.2),
                    axs.set_xlabel('Confidence'),
                    axs.set_ylabel('Accuracy'),
                    axs.set_title('Reliability Diagram'),
                    axs.legend(),
                    axs.grid(True)
                ),
                'filename': plot_names['Reliability_Diagram']
            },
            'Confidence_Histogram': {
                'data': confidences,
                'plot_func': lambda axs: (
                    axs.hist(confidences, bins=num_bins, range=(0, 1), edgecolor='black'),
                    axs.set_xlabel('Confidence'),
                    axs.set_ylabel('Number of Samples'),
                    axs.set_title('Histogram of Confidences'),
                    axs.grid(True)
                ),
                'filename': plot_names['Confidence_Histogram']
            },
            'Confidence_vs_Accuracy': {
                'data': (bin_centers, bin_confidences, bin_accuracies),
                'plot_func': lambda axs: (
                    axs.bar(bin_centers - 0.02, bin_confidences, width=0.04, label='Confidence', alpha=0.7),
                    axs.bar(bin_centers + 0.02, bin_accuracies, width=0.04, label='Accuracy', alpha=0.7),
                    axs.set_xlabel('Confidence Bins'),
                    axs.set_ylabel('Value'),
                    axs.set_title('Confidence vs Accuracy per ECE Bin'),
                    axs.legend(),
                    axs.grid(True)
                ),
                'filename': plot_names['Confidence_vs_Accuracy']
            },
            'Samples_per_Bin': {
                'data': (bin_centers, samples_in_bin),
                'plot_func': lambda axs: (
                    axs.bar(bin_centers, samples_in_bin, width=0.04, edgecolor='black'),
                    axs.set_xlabel('Confidence Bins'),
                    axs.set_ylabel('Number of Samples'),
                    axs.set_title('Number of Samples per ECE Bin'),
                    axs.grid(True)
                ),
                'filename': plot_names['Samples_per_Bin']
            }
        }
        for plot_name, config in plots.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            config['plot_func'](ax)

            if save:
                save_file = os.path.join(save_path, config['filename'])
                plt.savefig(save_file)
                print(f"{plot_name} saved to {save_file}")
            if show_plot:
                plt.show()
            else:
                plt.close()
                
    def run_laplace(self, subset_of_weights = "last_layer",hessian_structure = "kron"):
        '''
        For running Laplace approximation on the model
        Input : subset_of_weights : "all" or "last_layer"
                hessian_structure : "kron" , "diag" or "full"
                I would reccomend to not run FULL hessian as it is very computationally expensive
        '''
        print("Running Laplace Approximation")
        self.laplace = Laplace(self.model, 'classification', subset_of_weights=subset_of_weights, hessian_structure=hessian_structure)
        print(f"Laplace Approximation Done with {subset_of_weights} weights and {hessian_structure} hessian structure")
        print("Fitting the Laplace Approximation")
        self.laplace.fit(self.data_loader)
        print("Laplace Approximation Fitted")
        print("Optimizing the Prior Precision")
        self.laplace.optimize_prior_precision()
        print("Prior Precision Optimized")
        
  
    def return_BBP_variances(self):
        if self.ensamble == False:
            return extract_variances(self.model)
        else:
            raise NotImplementedError
        
    def return_BBP_variances_dataframe(self):
        variances = self.return_BBP_variances()
        return variances_to_dataframe(variances)
        
    def print_variances(self, number_to_show = 10):
        variances = self.return_BBP_variances()
        variances = variances_to_dataframe(variances)
        print(variances.head(number_to_show))
        
    def image_output_variance(self, image, ground_truth=None, number_of_samples=100, show_image=False, save_image=False, save_path=None):
        image = image.to(self.device)
        list_of_sampled_logits = []
        list_of_aleatoric_variances = []
        with torch.no_grad():
            if not self.ensamble:
                for _ in range(number_of_samples):
                    model_outputs = self.model(image)
                    if isinstance(model_outputs, (tuple, list)):
                        mean = model_outputs[0]
                        log_var = model_outputs[1]
                        sigma = torch.exp(0.5 * log_var)
                        epsilon = torch.randn_like(mean)
                        sampled_logits = mean + epsilon * sigma
                        aleatoric_variance = sigma.pow(2).cpu().numpy().reshape(-1)
                        list_of_aleatoric_variances.append(aleatoric_variance)
                    else:
                        sampled_logits = model_outputs
                    list_of_sampled_logits.append(sampled_logits)
            else:
                for tmp in self.model:
                    model_outputs = tmp(image)
                    if isinstance(model_outputs, (tuple, list)):
                        mean = model_outputs[0]
                        log_var = model_outputs[1]
                        sigma = torch.exp(0.5 * log_var)
                        epsilon = torch.randn_like(mean)
                        sampled_logits = mean + epsilon * sigma
                        aleatoric_variance = sigma.pow(2).cpu().numpy().reshape(-1)
                        list_of_aleatoric_variances.append(aleatoric_variance)
                    else:
                        sampled_logits = model_outputs
                    list_of_sampled_logits.append(sampled_logits)
            avg_outputs = torch.mean(torch.stack(list_of_sampled_logits), dim=0)
            _, predicted = torch.max(avg_outputs, 1)
            probabilities = torch.softmax(avg_outputs, dim=1)
            confidence_score = probabilities[0][predicted.item()].item()

            for i in range(len(list_of_sampled_logits)):
                list_of_sampled_logits[i] = torch.softmax(list_of_sampled_logits[i], dim=1).cpu().numpy().reshape(-1)
            list_of_logits = pd.DataFrame(list_of_sampled_logits)
            probabilities = pd.DataFrame(probabilities.cpu().numpy())

            mean_probs = np.mean(list_of_logits, axis=0)
            std_probs = np.std(list_of_logits, axis=0)
            ratio = std_probs / mean_probs

            if list_of_aleatoric_variances:
                mean_aleatoric_variance = np.mean(list_of_aleatoric_variances, axis=0)
            else:
                list_of_aleatoric_variances = None
                mean_aleatoric_variance = None

            if show_image:
                show_image_confidence(image.squeeze(0).cpu(), predicted.item(), ground_truth, confidence_score, self.image_paramters, save_image, save_path)

        return {
            'predicted_class': predicted.item(),
            'confidence_score': confidence_score,
            'probabilities': probabilities,
            'sampled_probabilities': list_of_logits,
            'mean_probabilities': mean_probs,
            'std_probabilities': std_probs,
            'uncertainty_ratio': ratio,
            'list_of_aleatoric_variances': list_of_aleatoric_variances,
            'mean_aleatoric_variance': mean_aleatoric_variance
        } 
