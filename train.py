import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import pydicom

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

#import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score, auc, log_loss

import os
import copy

from tqdm.notebook import tqdm
import tqdm as tq
import time

from EarlyStopping import EarlyStopping
import torch.optim as optim

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
        
        for phase in ["train", "dev", "test"]:
            losses = results[fold].results[phase].losses
            epoch_losses = results[fold].results[phase].epoch_losses
            epoch_scores = results[fold].results[phase].epoch_scores
            lr_rates = results[fold].results[phase].learning_rates
            precision = results[fold].results[phase].precision
            memory = results[fold].results[phase].memory
            tmp = results[fold].results[phase].time

            
            save_as_csv(losses, phase + "_losses", base_dir)
            save_as_csv(epoch_losses, phase + "_epoch_losses", base_dir)
            save_as_csv(epoch_scores, phase + "_epoch_scores", base_dir)
            save_as_csv(lr_rates, phase + "epoch_lr_rates", base_dir)
            save_as_csv(precision, phase + "epoch_precision", base_dir)
            save_as_csv(memory, phase + "epoch_memory", base_dir)
            save_as_csv(tmp, phase + "epoch_time", base_dir)


class ResultsBean:
    
    def __init__(self):
        
        self.precision = []
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
        self.test_results = ResultsBean()
        self.results = {"train": self.train_results,
                        "dev": self.dev_results,
                        "test": self.test_results}

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
                 LR,
                 MOMENTUM,
                 W_DECAY,
                 find_lr):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    phases = dataloaders_dict.keys()

    t0 = time.time()

    early_stop = EarlyStopping(patience)
    break_training = False
    best_acc = 0

    for epoch in range(num_epochs):
        
        for phase in phases:
            train_loss = 0
            correct = 0
            total = 0
            running_loss = 0
            
            dataloader = dataloaders_dict[phase]
            dataloader_iterator = tqdm(dataloader, total=int(len(dataloader)))
            
            if phase=="train":
                model.train()
            else:
                model.eval()
                
   
            for counter, data in enumerate(dataloader_iterator):
                image_input = data[0]
                target_input = data[1]
                
                image_input = image_input.to(device, dtype=torch.float)
                target_input = target_input.to(device, dtype=torch.long)
    
                optimiser.zero_grad()
                
                raw_output = model(image_input) 
                _, preds = raw_output.max(1)

                total += target_input.size(0)
                correct += preds.eq(target_input).sum().item()
                
                
                
              
                batch_size = target_input.shape[0]
                
                
                loss = criterion(raw_output, target_input)
                del raw_output
                # redo the average over mini_batch
                running_loss += (loss.item() * batch_size)
    
                # save averaged loss over processed number of batches:
                processed_loss = running_loss / ((counter+1) * batch_size)
                results.results[phase].losses.append(processed_loss)
                
                if phase == 'train':
                    loss.backward()
                    optimiser.step()
                    # if scheduler is not None:
                    #     scheduler.step()

            if phase == 'train':
                # loss.backward()
                # optimiser.step()
                if scheduler is not None:
                    scheduler.step()

            precision = 100.*correct/total
            score  = precision
           

            
            results.results[phase].epoch_scores.append(score)
            results.results[phase].memory.append(   convert_bytes(torch.cuda.max_memory_allocated()))
            results.results[phase].time.append(time.time() - t0)
            results.results[phase].precision.append(precision)
            results.results[phase].learning_rates.append(optimiser.state_dict()["param_groups"][0]["lr"])
            
            # average over all samples to obtain the epoch loss
            epoch_loss = running_loss / len(dataloader.dataset)
            results.results[phase].epoch_losses.append(epoch_loss)
            
            print("fold: {}, epoch: {}, phase: {}, e-loss: {}, prec: {}".format(
                fold_num, epoch, phase, epoch_loss, score))
            
            if not find_lr:
                if phase == "dev":
                    early_stop(-score, model)
                    if not early_stop.early_stop :
                      if precision >= best_acc:
                        best_model = {'state' : copy.deepcopy(model.state_dict()),
                                      'epoch' : epoch}
                        best_acc = precision

                    else:
                      if not scheduler.end:
                        early_stop.early_stop = False
                        early_stop.counter = 0
                        scheduler.update_lr()
                        print("Model hasn't improved for {} epochs. LR updated.".format(patience))
                        model.load_state_dict(best_model['state'])
                        optimizer = optim.SGD(model.parameters(), lr=LR,
                                  momentum=MOMENTUM, weight_decay=W_DECAY)
                      else:
                        print("Model hasn't improved for {} epochs. Training finished.".format(patience))
                        break_training = True
                        break


        if break_training:
          break

               
    # load best model weights
    if not find_lr and "test" not in phases:
        model.load_state_dict(best_model['state'])
    
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
          LR,
          MOMENTUM,
          W_DECAY,
          find_lr=False,
          results = None):
    
    if results == None:
      single_results = Results(fold_num, model)
    else: 
      single_results = results
    
    single_results = run_training(model,
                                  criterion,
                                  optimiser,
                                  num_epochs,
                                  dataloaders_dict,
                                  fold_num,
                                  scheduler,
                                  patience,
                                  single_results, 
                                  LR,
                                  MOMENTUM,
                                  W_DECAY,
                                  find_lr=find_lr)
       
    return single_results





def convert_bytes(size, isbytes = True):
  if isbytes :
    b = 1000.0
  else:
    b = 1024.0    
  for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
     if size < b:
       return "%3.1f %s" % (size, x)
     size /= b

  return size