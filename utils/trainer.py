import numpy as np
import torch
from sklearn import metrics

from utils.misc import plot_figures

def train_epoch(model, optimizer, criterion, scheduler, train_data, train_labels, test_data, test_labels, ep):
    model.train()
    run_train_acc, run_test_acc = 0.0, 0.0
    run_train_loss, run_test_loss = 0.0, 0.0
    for it, batch in enumerate(train_data):
        optimizer.zero_grad()
        batch = batch.squeeze()
        batch = batch.reshape(-1, batch.shape[-1]) # reshape for patch_embeddings
        batch = torch.tensor(batch, dtype=torch.float32).cuda()
        # batch = [torch.tensor(batch_it, dtype=torch.float32).cuda() for batch_it in batch] # ABMIL_fusion
        train_logits = model(batch)[0]
        loss = criterion(train_logits, torch.tensor(train_labels[it]).cuda())  # CE loss
        loss.backward()  # Gradient calculation
        optimizer.step()  # Gradient propagation
        train_pred = train_logits.softmax(axis=0).argmax(axis=0)
        run_train_acc += (train_pred == train_labels[it]).item()  # Train ACC
        run_train_loss += loss.item()
    train_acc = run_train_acc / len(train_data)
    train_loss = run_train_loss / len(train_data)

    scheduler.step()
    model.eval()
    with torch.no_grad():
        for it, batch in enumerate(test_data):
            batch = batch.squeeze()
            batch = batch.reshape(-1, batch.shape[-1]) # reshape for patch_embeddings
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            # batch = [torch.tensor(batch_it, dtype=torch.float32).cuda() for batch_it in batch]  # ABMIL_fusion
            test_logits = model(batch)[0]
            test_loss = criterion(test_logits, torch.tensor(test_labels[it]).cuda())
            test_pred = test_logits.softmax(axis=0).argmax(axis=0)
            run_test_acc += (test_pred == test_labels[it]).item()
            run_test_loss += test_loss.item()
    test_acc = run_test_acc / len(test_data)
    test_loss = run_test_loss / len(test_data)

    print('-------------------------------------------------------------------------')
    print(f'Epoch {ep + 1} \t Training Loss = {train_loss:.4f} \t Validation Loss = {test_loss:.4f}')
    print(f'         \t Training Acc  = {train_acc:.4f} \t Validation Acc  = {test_acc:.4f}')
    print('-------------------------------------------------------------------------')

    return train_acc, test_acc, train_loss, test_loss

def train_model(model, optimizer, criterion, scheduler, train_data, train_labels, test_data, test_labels, epochs, run_name, save_dir=None):
    train_acc_epoch, test_acc_epoch, train_loss_epoch, test_loss_epoch = [], [], [], []
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_labels)) # Shuffle data in each epoch
        train_data, train_labels = [train_data[idx] for idx in indices], [train_labels[idx] for idx in indices]
        metrics = train_epoch(model,optimizer,criterion,scheduler,train_data,train_labels,test_data,test_labels,epoch)
        train_acc_epoch.append(metrics[0])
        test_acc_epoch.append(metrics[1])
        train_loss_epoch.append(metrics[2])
        test_loss_epoch.append(metrics[3])
    if save_dir is not None:
        plot_figures(train_acc_epoch, test_acc_epoch, train_loss_epoch, test_loss_epoch, run_name, save_dir)
    
def validate_model(model, test_data, test_labels):
    model.eval()
    list_test_pred = []
    list_test_pred_proba = []
    with torch.no_grad():
        for it, batch in enumerate(test_data):
            batch = batch.squeeze()
            batch = batch.reshape(-1, batch.shape[-1]) # patch-embeddings
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            test_logits = model(batch)[0]
            test_pred = test_logits.softmax(axis=0).argmax(axis=0).item()
            list_test_pred.append(test_pred)
            list_test_pred_proba.append(test_logits.softmax(axis=0).cpu().numpy())
    list_test_pred = np.stack(list_test_pred)
    list_test_pred_proba = np.stack(list_test_pred_proba)
    conf_mx = metrics.confusion_matrix(test_labels, list_test_pred)
    return conf_mx, test_labels, list_test_pred_proba

def validate_model_v2(model, test_data, test_labels):
    model.eval()
    list_test_pred, list_test_prob = [], []
    with torch.no_grad():
        for it, batch in enumerate(test_data):
            batch = torch.tensor(batch, dtype=torch.float32).cuda()
            test_logits = model(batch)[0]
            test_prob = test_logits.softmax(axis=0)
            test_pred = test_prob.argmax(axis=0).item()
            list_test_prob.append(test_prob.cpu().numpy())
            list_test_pred.append(test_pred)
    list_test_pred = np.stack(list_test_pred)
    list_test_prob = np.stack(list_test_prob)
    conf_mx = metrics.confusion_matrix(test_labels, list_test_pred)
    return conf_mx, list_test_pred, list_test_prob