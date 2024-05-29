# Imports
import os
import json
from time import time
from datetime import datetime
from matplotlib import pyplot as plt

class LoggingHandler():
    def __init__(self, save_dir):
        # idea, just have a dictionary with the stats that you save at the end of each
        # epoch. You save it down to json at the end. On start, you create a new logging
        # text file where you have info about the running script. IF there doesn't exist a
        # file 'checkpoint_latest_stats.json' then create one. This is where you will load
        # up the stats from the previous run. If there is no such file, then you start
        # from scratch.

        self.per_epoch_stats = dict()
        self.save_dir = save_dir
        self.curr_epoch = -1
        self.curr_epoch_stats = dict()


    def save_stats(self):
        with open(os.path.join(self.save_dir, 'checkpoint_latest_stats.json'), 'w') as json_file:
            json.dump(self.per_epoch_stats, json_file)

    def load_stats(self):
        with open(os.path.join(self.save_dir, 'checkpoint_latest_stats.json'), 'r') as json_file:
            self.per_epoch_stats = json.load(json_file)

    def log_metric(self, key, value, epoch):
        assert self.curr_epoch == epoch
        if key not in self.curr_epoch_stats:
            self.curr_epoch_stats[key] = [value]
        else:
            self.curr_epoch_stats[key].append(value)

    def log(self, line):
        with open(self.loggerName, 'a') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {line}\n')

    def start_new_epoch(self, epoch):
        if self.curr_epoch >= 0:
            self.save_stats()
            self.per_epoch_stats[self.curr_epoch] = self.curr_epoch_stats
        self.curr_epoch = epoch
        self.curr_epoch_stats = {}

        self.curr_epoch_stats['epoch_start'] = time()

    def end_current_epoch(self, epoch):
        assert epoch == self.curr_epoch
        end_time = time()
        self.curr_epoch_stats['epoch_end'] = end_time
        self.curr_epoch_stats['epoch_time'] = end_time - self.curr_epoch_stats['epoch_start']

        self.per_epoch_stats[self.curr_epoch] = self.curr_epoch_stats
        self.save_stats()

    def setup_logger(self):
        # check if the file exists
        if os.path.exists(os.path.join(self.save_dir, 'checkpoint_latest_stats.json')):
            self.load_stats()
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_stats()

        # create a logging file based on datetime
        now = datetime.now()
        self.loggerName = os.path.join(self.save_dir, f'training_{now.strftime("%Y%m%d_%H%M%S")}.log')
        
        with open(self.loggerName, 'w') as f:
            f.write(f"==============================\n")
            f.write(f"Initialised Logger at {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Happy Logging!\n")
            f.write(f"==============================\n")
    
    def plot_stats(self):
        self.load_stats()  # Make sure that the data is current

        # Extract data for plotting
        epochs = []
        dice_loss = []
        ce_loss = []
        val_dice_loss = []
        val_ce_loss = []
        epoch_time = []

        for key, value in self.per_epoch_stats.items():
            epochs.append(int(key))
            dice_loss.append(sum(value['dice_loss']) / len(value['dice_loss']))
            ce_loss.append(sum(value['ce_loss']) / len(value['ce_loss']))
            val_dice_loss.append(sum(value['val_dice_loss']) / len(value['val_dice_loss']))
            val_ce_loss.append(sum(value['val_ce_loss']) / len(value['val_ce_loss']))
            epoch_time.append(value['epoch_time'])

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Loss plot
        ax1.plot(epochs, dice_loss, label='Dice Loss (Train)', marker='o')
        ax1.plot(epochs, ce_loss, label='CE Loss (Train)', marker='o')
        ax1.plot(epochs, val_dice_loss, label='Dice Loss (Validation)', marker='o')
        ax1.plot(epochs, val_ce_loss, label='CE Loss (Validation)', marker='o')

        ax1.set_title('Losses Over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Epoch time plot
        ax2.plot(epochs, epoch_time, label='Epoch Time', color='orange', marker='o')

        ax2.set_title('Epoch Time Over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Epoch Time (seconds)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'progress.png'))
        plt.clf()