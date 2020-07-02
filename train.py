import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import pydicom

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

#import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score, auc

import os
import copy

from tqdm.notebook import tqdm
import time

from EarlyStopping import EarlyStopping

def save_as_csv(series, name, path):
    df = pd.DataFrame(index=np.arange(len(series)), data=series, columns=[name])
    output_path = path + name + ".csv"
    df.to_csv(output_path, index=False)

def save_results(results, foldername):
    for fold in results.keys():
        
        base_dir = foldername + "/fold_" + str(fold) + "/"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # save the model for inference
        model = results[fold].model
        model_path = base_dir + "model.pth"
        torch.save(model.state_dict(), model_path)
        
        for phase in ["train", "dev"]:
            losses = results[fold].results[phase].losses
            epoch_losses = results[fold].results[phase].epoch_losses
            epoch_scores = results[fold].results[phase].epoch_scores
            lr_rates = results[fold].results[phase].learning_rates
            f1_scores = results[fold].results[phase].f1_scores
            precision = results[fold].results[phase].precision
            recall = results[fold].results[phase].recall
            memory = results[fold].results[phase].memory
            tmp = results[fold].results[phase].time

            
            save_as_csv(losses, phase + "_losses", base_dir)
            save_as_csv(epoch_losses, phase + "_epoch_losses", base_dir)
            save_as_csv(epoch_scores, phase + "_epoch_scores", base_dir)
            save_as_csv(lr_rates, phase + "_lr_rates", base_dir)
            save_as_csv(f1_scores, phase + "_f1_scores", base_dir)
            save_as_csv(precision, phase + "_precision", base_dir)
            save_as_csv(recall, phase + "_recall", base_dir)
            save_as_csv(memory, phase + "_memory", base_dir)
            save_as_csv(tmp, phase + "_time", base_dir)


class ResultsBean:
    
    def __init__(self):
        
        self.precision = []
        self.recall = []
        self.f1_scores = []
        self.losses = []
        self.learning_rates = []
        self.epoch_losses = []
        self.epoch_scores = []
        self.memory = []
        self.time = []

class Results:
    
    def __init__(self, fold_num, model=None):
        self.model = model
        self.fold_num = fold_num
        self.train_results = ResultsBean()
        self.dev_results = ResultsBean()
        self.results = {"train": self.train_results,
                        "dev": self.dev_results}

def get_lr_search_scheduler(optimiser, min_lr, max_lr, max_iterations):
    # max_iterations should be the number of steps within num_epochs_*epoch_iterations
    # this way the learning rate increases linearily within the period num_epochs*epoch_iterations 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser, 
                                               base_lr=min_lr,
                                               max_lr=max_lr,
                                               step_size_up=max_iterations,
                                               step_size_down=max_iterations,
                                               mode="triangular")
    
    return scheduler


def get_scheduler(optimiser, min_lr, max_lr, stepsize):
    # suggested_stepsize = 2*num_iterations_within_epoch
    stepsize_up = np.int(stepsize/2)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimiser,
                                               base_lr=min_lr,
                                               max_lr=max_lr,
                                               step_size_up=stepsize_up,
                                               step_size_down=stepsize_up,
                                               mode="triangular")
    return scheduler

def run_training(model,
                 criterion,
                 optimiser,
                 num_epochs,
                 dataloaders_dict,
                 fold_num,
                 scheduler,
                 patience,
                 results,
                 find_lr):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    phases = ["train", "dev"]
    best_auc = 0
    patience_counter = 0
    epsilon = 1e-7
    t0 = time.time()

    early_stop = EarlyStopping(patience)
    
    for epoch in range(num_epochs):
        
        for phase in phases:
            
            dataloader = dataloaders_dict[phase]
            dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))
            
            if phase=="train":
                model.train()
            else:
                model.eval()
                
            all_probas = np.zeros(len(dataloader)*dataloader.batch_size)
            all_targets = np.zeros(len(dataloader)*dataloader.batch_size)   
            running_loss = 0.0
            running_true_positives = 0
            running_false_positives = 0
            running_false_negatives = 0
            
                      
            for counter, data in enumerate(dataloader_iterator):
                image_input = data["image"]
                target_input = data["target"]
                
                image_input = image_input.to(device, dtype=torch.float)
                target_input = target_input.to(device, dtype=torch.long)
    
                optimiser.zero_grad()
                
                raw_output = model(image_input) 
                pred_probas = F.softmax(raw_output, dim=1)
                _, preds = torch.max(pred_probas, 1)
                
                
                running_true_positives += (preds*target_input).sum().cpu().detach().numpy()
                running_false_positives += ((1-target_input)*preds).sum().cpu().detach().numpy()
                running_false_negatives += (target_input*(1-preds)).sum().cpu().detach().numpy()

                precision = running_true_positives/ (running_true_positives + running_false_positives + epsilon)
                recall = running_true_positives/ (running_true_positives + running_false_negatives + epsilon)
                f1_score = 2*precision*recall/ (precision+recall+epsilon) 
                
                
                results.results[phase].learning_rates.append(optimiser.state_dict()["param_groups"][0]["lr"])
                results.results[phase].precision.append(precision)
                results.results[phase].recall.append(recall)
                results.results[phase].f1_scores.append(f1_score)
                        
                batch_size = dataloader.batch_size
                all_targets[(counter*batch_size):((counter+1)*batch_size)] = target_input.cpu().detach().numpy()
                all_probas[(counter*batch_size):((counter+1)*batch_size)] = pred_probas.cpu().detach().numpy()[:,1]
                
                loss = criterion(raw_output, target_input)
                # redo the average over mini_batch
                running_loss += (loss.item() * batch_size)
    
                # save averaged loss over processed number of batches:
                processed_loss = running_loss / ((counter+1) * batch_size)
                results.results[phase].losses.append(processed_loss)
                
                if phase == 'train':
                    loss.backward()
                    optimiser.step()
                    if scheduler is not None:
                        scheduler.step()
            
            epoch_auc_score = roc_auc_score(all_targets, all_probas)
            results.results[phase].epoch_scores.append(epoch_auc_score)
            results.results[phase].memory.append(torch.cuda.max_memory_allocated())
            results.results[phase].time.append(time.time() - t0)
                
            
            # average over all samples to obtain the epoch loss
            epoch_loss = running_loss / len(dataloader.dataset)
            results.results[phase].epoch_losses.append(epoch_loss)
            
            print("fold: {}, epoch: {}, phase: {}, e-loss: {}, e-auc: {}".format(
                fold_num, epoch, phase, epoch_loss, epoch_auc_score))
            
            if not find_lr:
                if phase == "dev":
                    # if epoch_auc_score >= best_auc:
                    #     best_auc = epoch_auc_score
                    #     best_model_wts = copy.deepcopy(model.state_dict())
                    # else:
                    #     patience_counter += 1
                    #     if patience_counter == patience:
                    #         print("Model hasn't improved for {} epochs. Training finished.".format(patience))
                    #         break
                    early_stop(-epoch_auc_score, model)
                    if not early_stop.early_stop:
                      best_model_wts = copy.deepcopy(model.state_dict())
                    else:
                      print("Model hasn't improved for {} epochs. Training finished.".format(patience))
                      break

               
    # load best model weights
    if not find_lr:
        model.load_state_dict(best_model_wts)
    
    results.model = model
    return results


def train(model,
          criterion,
          optimiser,
          num_epochs,
          dataloaders_dict,
          fold_num,
          scheduler,
          patience,
          find_lr=False):
    
    single_results = Results(fold_num, model)
    
    single_results = run_training(model,
                                  criterion,
                                  optimiser,
                                  num_epochs,
                                  dataloaders_dict,
                                  fold_num,
                                  scheduler,
                                  patience,
                                  single_results, 
                                  find_lr=find_lr)
       
    return single_results