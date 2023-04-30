import numpy as np

class MultiplicativeDecayWithReduceOrRestartOnPlateau:
    def __init__(self, optimizer, lr_init, lr_decay, lr_max, f_reduction, 
                 f_restart, patience, min_loss_improvement, 
                 no_of_restarts_init, increment_no_of_restarts):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_max = lr_max
        self.f_reduction = f_reduction
        self.f_restart = f_restart
        self.patience = patience
        self.min_loss_improvement = min_loss_improvement
        self.no_of_restarts_init = no_of_restarts_init
        self.increment_no_of_restarts = increment_no_of_restarts
        
        self.loss_best = np.Inf
        self.plateau_epochs = 0
        self.reductions_counter = 0
        self.just_restarted = False
        self.set_par("initial_lr", self.lr_init)
        self.set_par("lr", self.lr_init)
        
    def get_par(self, name):
        for group in self.optimizer.param_groups: par = group[name]
        return par
    
    def set_par(self, name, value):
        for group in self.optimizer.param_groups: group[name] = value
        
    def on_loss_improvement(self, loss_cur):
        self.loss_best = loss_cur
        self.plateau_epochs = 0
    
    def on_loss_deterioration(self):
        self.plateau_epochs += 1
    
    def multiplicative_decay_lr(self, lr):
        return lr * self.lr_decay
    
    def on_plateau(self, lr):
        if self.reductions_counter >= self.no_of_restarts_init:
            lr_new = self.restart_lr()
        else:
            lr_new = self.steep_reduce_lr(lr)
            return lr_new

    def steep_reduce_lr(self, lr):
        lr_new = lr * self.f_reduction
        self.plateau_epochs = 0
        self.reductions_counter += 1
        return lr_new
    
    def restart_lr(self):
        self.lr_init *= self.f_restart
        lr_new = self.lr_init
        if lr_new > self.lr_max:
            lr_new = self.lr_max
        self.plateau_epochs = 0
        self.reductions_counter = 0
        self.no_of_restarts_init += self.increment_no_of_restarts
        self.just_restarted = True
        return lr_new
    
    def step(self, metrics):
        lr_cur = self.get_par("lr")
        loss_cur = float(metrics)
        
        if 1 - (loss_cur / self.loss_best) > self.min_loss_improvement:
            self.on_loss_improvement(loss_cur)
        else:
            self.on_loss_deterioration() 
        lr_new = self.multiplicative_decay_lr(lr_cur)
        
        if self.just_restarted:
            self.loss_best = loss_cur
            self.just_restarted = False
            
        if self.plateau_epochs >= self.patience:
            lr_new = self.on_plateau(lr_cur)
        self.set_par("lr", lr_new)
     