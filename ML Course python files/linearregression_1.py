import torch

#Linear regression model
# linear model f = w * x + b
# f = 2 * x

X= torch.tensor([1,2,3,4,5,6,7,8], dtype=torch.float32)
Y= torch.tensor([2,4,6,8,10,12,14,16], dtype=torch.float32)

w= torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

#model output
def forward(x):
    return w*x

#Mean Squared Error
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

X_test = 9

print(f'Prediction before training: f({X_test})={forward(X_test).item():.3f}')

#Hyperparameters
learning_rate = 0.02
n_epochs = 15

for epoch in range(n_epochs):
    #predictions of the forward pass
    y_pred = forward(X)

    #loss calculation
    l = loss(Y,y_pred)

    #Calculate gradient descent
    l.backward()

    #update weights
    with torch.no_grad():
        w-=learning_rate*w.grad

    w.grad.zero_()

    print(f'epoch{epoch+1}:w = {w.item():.3f},loss = {l.item():.3f}')

print(f'Prediction post-training: f({X_test}) = {forward(X_test).item():.3f}')
