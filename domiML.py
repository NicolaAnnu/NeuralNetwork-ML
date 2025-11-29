# ML
import numpy as np
import scipy as sp
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    ds = s *(1-s)
    return s , ds

def relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return (x>0).astype(float)




def loss_function(output, x):
    return -(x*np.log(output) + (1-x)*np.log(1-output))

class Layer:
    def __init__(self, n_neurons, n_neurons_back,last_layer=False):

        self.n_neurons = n_neurons
        self.last_layer = last_layer

        self.W = np.random.randn(self.n_neurons, n_neurons_back) * np.sqrt(2/n_neurons_back)    # matrice dei pesi,
        self.b = np.zeros((self.n_neurons, 1))   
        print(np.size(self.W))  
        print(np.size(self.b))  
        # variabili che verranno usate successivamente   
        self.x = None       #ingressi 
        self.net = None         #integrazione
        self.out= None    

        self.dW = None
        self.db = None               



    def forward(self, x):  #funzione per forward propagation
        self.x = x
        self.net = self.W @ x + self.b

        if self.last_layer:
            self.out = sigmoid(self.net)[0]  #applico funzione di attivazione sigmoide per l'ultimo layer
        else:
            self.out = relu(self.net)   #applico funzione di attivazione f
        return self.out  #funzione per forward propagation
    
    def backward_last(self, Y_batch):
        delta = (self.out - Y_batch)
        self.dW = delta @ self.x.T
        self.db = np.sum(delta, axis=1, keepdims=True)

        return delta

    def backward(self, delta_succ, w_succ):  #funzione per backward propagation, input data di layer precedente
        
        delta = (w_succ.T @ delta_succ)*relu_deriv(self.net)
        self.dW = delta @ self.x.T
        self.db = np.sum(delta, axis = 1, keepdims = True)

        return delta #tanti quanti sono i neuroni del layer
    
    def update_w (self, eta):
        batch_size = self.x.shape[1]
        self.W -= eta * (self.dW/batch_size)
        self.b -= eta * (self.db/batch_size)




class Network:
    def __init__(self, net_index, eta):

        self.net_index = net_index
        self.eta = eta
        self.layers = []    #creiamo gli oggetti layer

        for i in range(1, len(net_index)):

            last = (i == len(net_index)-1)
            self.layers.append(Layer(net_index[i], net_index[i-1], last_layer=last))

        #variabili per il training
        self.current_loss = 0
        

        

    def net_forward(self, x):
        out = x
        for L in self.layers:   #si fa il forward su tutti i layer sequenzialmente

            out = L.forward(out) # gli passo come input l'out del precedente, dovro' inizializzare out come l'input esterno
            
        return out
        
            #CALCOLO LOSS
    def net_Loss(self, out, Y_batch):

        batch_loss = np.mean(np.sum(loss_function(out, Y_batch), axis=0))
        self.current_loss = batch_loss
        return batch_loss

            


            #PARTE BACKWARD
    def net_backward(self, Y_batch):
        delta = None
        for i in range(len(self.layers)-1, -1 , -1): 
            
            if i == len(self.layers)-1:
                # Ultimo layer: chiama backward_first passando il Target Reale
                delta = self.layers[i].backward_last(Y_batch)
            else:
                # Layer nascosti: chiama backward passando delta e W del layer successivo (i+1)
                delta = self.layers[i].backward(delta, self.layers[i+1].W)


    def net_w_update(self):
        for layer in self.layers:
            layer.update_w(self.eta)

#ora faccio la funzione del main
    


train_file = 'monks_train1.csv'  # il file di train
#hyperparametri
eta = 0.1  #non ho capito se deve essere positivo o negativo
batch_size = 20
epoch = 200
df = pd.read_csv(train_file, sep=",", usecols = range(7), skiprows = 1)  # carica dati in un dataframe il train data

y = df.iloc[ : , 0].to_numpy() # la prima colonna in un vettore numpy
x = df.iloc[ : , 1:7].to_numpy() # le altre 5 colonne in una matrice numpy

x, x_valid, y, y_valid = train_test_split(x,y, test_size = 0.2)

y = y.reshape(-1,1)
y_valid = y_valid.reshape(-1,1)


encoder = OneHotEncoder(sparse_output=False)
x = encoder.fit_transform(x)
x_valid = encoder.transform(x_valid)

# rirmalizzazione e preparazione dati
x = x - 0.5
x_valid = x_valid - 0.5

x = x.T  # piu facile lavorare con vettori colonna
y = y.T
x_valid = x_valid.T
y_valid = y_valid.T


loss_train = []
loss_valid = []

net_index = [x.shape[0], 12,15, y.shape[0]]

num_samples = x.shape[1]
total_batches = num_samples // batch_size
num_samples_valid = max(1,num_samples // batch_size)



#per fare il figo controlliamo che non siano troppi pochi dati per fare anche una sola batch

    

network = Network(net_index, eta)

for e in range (epoch):   #ciclo delle epoche
    epoch_loss_accum = 0

    x_shuf = x
    y_shuf = y
    #perm = np.random.permutation(num_samples)
    #x_shuf = x[:, perm]
    #y_shuf = y[:, perm]


    for b in range(total_batches):    #ciclo delle batch
            
        start = batch_size * b
        end = start + batch_size
        X_batch = x_shuf[:, start : end]
        Y_batch = y_shuf[:, start : end]

        out = network.net_forward(X_batch)

        loss = network.net_Loss(out, Y_batch)
        epoch_loss_accum += loss
        network.net_backward(Y_batch)

        network.net_w_update()

    loss_train.append(epoch_loss_accum / total_batches)
    #adesso vediamo il validation
        
    out_val = network.net_forward(x_valid)

    val_loss = network.net_Loss(out_val, y_valid)

    loss_valid.append(val_loss)

plt.plot(loss_train, label= 'train')
plt.plot(loss_valid, label = 'valid')
plt.legend()
plt.show()




    

    


        