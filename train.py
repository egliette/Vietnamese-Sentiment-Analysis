import time
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import utils


def train(model, dataset, batch_size, optimizer, criterion, pad_idx, device):
    """
    @param model (RNN)
    @param dataset (IMDBDataset)
    @param batch_size (int)
    @param optimizer (torch.optim)
    @param criterion (torch.nn.modules.loss)
    @param pad_idx (int)
    @param device (torch.device)
    @return epoch_loss (float): model"s loss of this epoch
    @return epoch_acc (float): model"s accuracy of this epoch 
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in utils.batch_iterator(dataset, batch_size, pad_idx, device):
        optimizer.zero_grad()
        reviews, reviews_lengths = batch["reviews"]
        predictions = model(reviews, reviews_lengths).squeeze(1)
        loss = criterion(predictions, batch["sentiments"])
        acc = utils.binary_accuracy(predictions, batch["sentiments"])
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    batch_num = math.ceil(len(dataset) / batch_size)
    return epoch_loss / batch_num, epoch_acc / batch_num

def evaluate(model, dataset, batch_size, criterion, pad_idx, device):
    """
    @param model (RNN)
    @param dataset (IMDBDataset)
    @param batch_size (int)
    @param criterion (torch.nn.modules.loss)
    @param pad_idx (int)
    @param device (torch.device)
    @return epoch_loss (float): model"s loss of this epoch
    @return epoch_acc (float): model"s accuracy of this epoch 
    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in utils.batch_iterator(dataset, batch_size, pad_idx, device):
            reviews, reviews_lengths = batch["reviews"]
            predictions = model(reviews, reviews_lengths).squeeze(1)
            loss = criterion(predictions, batch["sentiments"])
            acc = utils.binary_accuracy(predictions, batch["sentiments"])

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    batch_num = math.ceil(len(dataset) / batch_size)
    return epoch_loss / batch_num, epoch_acc / batch_num

def main(config_fpath):
    config = utils.get_config(config_fpath)

    print("Create logs folder...")
    current_log_dir, state_dir = utils.create_logs_dir(config)
    print(f"The current log dir is {current_log_dir}")

    print("Creating vocabulary...")     
    vocab, word_embedding = utils.get_vocab_and_word2vec(config, 
                                                         current_log_dir)    
    pad_idx = vocab["<pad>"]

    print("Loading dataset...")
    train_dataset, valid_dataset, test_dataset = utils.get_dataset(config,
                                                                   vocab,
                                                                   current_log_dir)
    
    print("Creating model...")
    model = utils.get_model(config, vocab, word_embedding)

    print("Creating optimizer and loss function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss().to(device)
    model = model.to(device)

    print("Creating SummaryWriter...")
    writer = utils.get_writer(log_dir=current_log_dir)

    print("Training...")
    n_epochs = config["train"]["n_epochs"]
    batch_size = config["train"]["batch_size"]
    save_epoch = config["train"]["save_epoch"]

    state_fpath = config["continue"]["state_fpath"]

    best_valid_loss = float("inf")

    if state_fpath is not None:
        checkpoint = torch.load(state_fpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch'] + 1
        print(f"Continue after epoch {begin_epoch}")
    else:
        begin_epoch = 0

    for epoch in range(begin_epoch, n_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_dataset, batch_size, 
                                    optimizer, criterion, pad_idx, device)
        valid_loss, valid_acc = evaluate(model, valid_dataset, batch_size, 
                                        criterion, pad_idx, device)
        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        writer.add_scalars(main_tag="loss",
                        tag_scalar_dict={"valid": valid_loss, 
                                            "train": train_loss},
                        global_step=epoch)
        writer.add_scalars(main_tag="accuracy",
                        tag_scalar_dict={"valid": valid_acc,
                                            "train": train_acc},
                        global_step=epoch)
        
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, f"{current_log_dir}/model.pt")

        if ((epoch + 1) % save_epoch) == 0:
            print(f"Saving state at epoch {epoch+1}")
            state_fpath = f"{state_dir}/epoch{epoch+1}.pt"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, state_fpath)

    print("Testing...")
    test_loss, test_acc = evaluate(model, test_dataset, batch_size, 
                               criterion, pad_idx, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese Sentiment Analyis model")

    parser.add_argument("--config", 
                        default="configs/config.yml", 
                        help="path to config file",
                        dest="config_fpath")
    
    args = parser.parse_args()

    main(**vars(args))
