import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn.functional as F
from torch.autograd import Variable

    
def compute_loss(X_batch, y_batch, model, device=1):
    '''Compute cross entropy loss'''
    X_batch = Variable(torch.FloatTensor(X_batch).cuda(device))
    y_batch = Variable(torch.LongTensor(y_batch).cuda(device))
    logits = model(X_batch)
    return F.cross_entropy(logits, y_batch).mean()

def train(model, train_loader, val_loader, num_epochs=100, lr=0.001, batch_size=20, device=1):
    '''Train and evaluate model'''
    train_loss, train_accuracy, train_loss_plot, train_acc_plot = [], [], [], []
    val_loss, val_accuracy, val_loss_plot, val_acc_plot = [], [], [], []
    
    for epoch in range(num_epochs):
        model.train(True) 
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for X_batch, y_batch in train_loader:
            loss = compute_loss(X_batch, y_batch, model, device=device)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss.append(loss.data.cpu().numpy()[0])
            
            logits = model(Variable(torch.FloatTensor(X_batch).cuda(device)))
            y_pred = logits.max(1)[1].data.cpu().numpy()
            y_batch = y_batch.numpy()
            train_accuracy.append(np.mean(y_batch == y_pred))
        if epoch%30==0:
            lr=lr/1.5
            
        model.train(False)
        for X_batch, y_batch in val_loader:
            logits = model(Variable(torch.FloatTensor(X_batch).cuda(device)))
            y_pred = logits.max(1)[1].data.cpu().numpy()
            y_batch = y_batch.numpy()
            val_accuracy.append(np.mean(y_batch == y_pred))
            loss = compute_loss(X_batch, y_batch, model)
            val_loss.append(loss.data.cpu().numpy()[0])
            
        if epoch==0:
            prev_min=100
        else: prev_min = min(val_loss_plot)  
            
        train_loss_plot.append(np.mean(train_loss[-len(train_loader) :]))
        train_acc_plot.append(np.mean(train_accuracy[-len(train_loader)  :]))
        val_loss_plot.append(np.mean(train_loss[-len(val_loader)  :]))
        val_acc_plot.append(np.mean(val_accuracy[-len(val_loader) :]))

        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        ax[0].set_title("Dynamic of loss", fontsize=15)
        ax[0].set_xlabel("Number of epoch", fontsize=13)
        ax[0].set_ylabel("loss", fontsize=13)
        ax[0].plot(train_loss_plot, label='train')
        ax[0].plot(val_loss_plot, label='val')
        ax[0].legend(fontsize=13)

        ax[1].set_title("Dynamic of accuracy", fontsize=15)
        ax[1].set_xlabel("Number of epoch", fontsize=13)
        ax[1].set_ylabel("accuracy", fontsize=13)
        ax[1].plot(train_acc_plot, label='train') 
        ax[1].plot(val_acc_plot, label='val') 
        ax[1].legend(fontsize=13)
        plt.show()

        if prev_min > min(val_loss_plot):
            torch.save(model.state_dict(), 'best_model.pt')
        
def test(model, test_loader, device=1):
    '''Compute accuracy score of model on test data'''
    
    model.train(False) 
    test_batch_acc = []
    for X_batch, y_batch in test_loader:
        logits = model(Variable(torch.FloatTensor(X_batch).cuda(device)))
        y_pred = logits.max(1)[1].data.cpu().numpy()
        y_batch = y_batch.numpy()
        test_batch_acc.append(np.mean(y_batch == y_pred))

    test_accuracy = np.mean(test_batch_acc)

    return test_accuracy
    
        
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std())
    x *= 0.2
    x += 0.5
    x = np.clip(x, 0, 1)
    im  = np.zeros((32,32,3))
    for k in range(3):
        im[:,:,k] = x[k,:,:] 
        
    return im

def plot_results(model, test_loader, device=1):
    fig, ax = plt.subplots(4, 5, figsize=(20, 15))
    for X_batch, y_batch in test_loader:
        logits = model(Variable(torch.FloatTensor(X_batch).cuda(device)))
        y_pred = logits.max(1)[1].data.cpu().numpy()
        if np.all(y_pred == y_batch.numpy()):
            continue
        for i in range(20):
                ax[i//5, i%5].imshow(deprocess_image(X_batch[i].numpy()))
                if y_pred[i] == 1:
                    ax[i//5, i%5].set_title('crocodile' , fontsize=15)
                else:
                    ax[i//5, i%5].set_title('clock', fontsize=15)
                ax[i//5, i%5].set_xticks([])
                ax[i//5, i%5].set_yticks([])
                
                if y_pred[i] != y_batch.numpy()[i]:
                    ax[i//5, i%5].title.set_color('red')
        plt.suptitle('Предсказания для тестовой выборки', fontsize=25, y=0.95)

        break