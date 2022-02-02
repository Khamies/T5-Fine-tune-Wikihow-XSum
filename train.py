
import torch
from datasets import load_metric
from utils import generate_summary

from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, train_loader, val_loader, model, tokenizer, optimizer, metric) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = SummaryWriter(flush_secs=5)
        self.rouge_metric = metric
        self.interval = 100


    def train(self, train_stats):

        for i, batch in enumerate(self.train_loader):
        

            source, source_mask, target, _ = batch['source'].squeeze(1).to(self.device), batch['source_mask'].to(self.device),\
                                                    batch['target'].squeeze(1), batch['target_mask'].to(self.device)
            
            target = torch.tensor(target)
            target[target[: ,:] == 0 ] = -100 # to let the model skip computing the loss of the zeros ( paddings)
            target = target.to(self.device)
       

            output = self.model(source, source_mask, labels = target, return_dict=True)

          
            mloss = output['loss']

            self.logger.add_scalar("mloss", mloss, global_step=i)
           
            
            self.optimizer.zero_grad()
            mloss.backward()
            self.optimizer.step()
            
            # For stats & visualization
            train_stats["loss"].append(mloss.item())
            
            # # Model evaluation
            # if i%self.interval==0:
            #     self.evaluate()
  
        self.logger.close()
        return train_stats
        
    def evaluate(self):

        with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
            

                    source, source_mask, target, target_mask = batch['source'].squeeze(1).to(self.device), batch['source_mask'].to(self.device),\
                                                            batch['target'].squeeze(1).to(self.device), batch['target_mask'].to(self.device)

                    target = torch.tensor(target)
                    target[target[: ,:] == 0 ] = -100
                    target = target.to(self.device)

                    
                    all_text = generate_summary(self.model, self.tokenizer, source, target)

                    # Rouge metric
                    rouge_sc = self.rouge_metric.compute(predictions=all_text["machine_text"], references=all_text["human_text"] )

                    self.logger.add_scalar("Rouge Precision", rouge_sc['rouge1'].mid[0], global_step=i)
                    self.logger.add_scalar("Rouge Recall", rouge_sc['rouge1'].mid[1], global_step=i)
                    self.logger.add_scalar("Rouge FScore", rouge_sc['rouge1'].mid[2], global_step=i)
            
