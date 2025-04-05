import torch
import torch.nn as nn

#Linear regression model
# linear model f = w * x + b
# f = 2 * x


# 0) Training samples
X= torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]], dtype=torch.float32)
Y= torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f"n_samples = {n_samples}, n_features = {n_features}")

X_test = torch.tensor([5],dtype=torch.float32)

#1
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        #vvvvv neural net layer implementation
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

input_size, output_size = n_features, n_features
model = LinearRegression(input_size,output_size)

print(f"Prediction before training: f({X_test.item()}) = {model(X_test).item():3f}")

#2 hyperparameters
learning_rate = 0.03
n_epochs = 400

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3 training
for epoch in range(n_epochs):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    w,b = model.parameters()
    print('epoch', epoch, ': w =', w[0][0].item(),'loss = ', l.item())

print(f'Predictions after training: f({X_test.item()}) = {model(X_test).item():.3f}')
