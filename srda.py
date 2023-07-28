import os
import torch
from torch import nn
from math import log
from sklearn.covariance import OAS

class StreamingRDA(nn.Module):
    """Pytorch implementation of Deep Streaming Regularized Discriminant Analysis.
    """

    def __init__(self, input_size, num_classes, test_batch_size=128, shrinkage_param=1e-4,
                 streaming_update_sigma=True,alpha=0.5,device=torch.device('cuda')):
        """Init function for the SRDA model

        Args:
            input_size (int): feature dimension
            num_classes (int): number of total classes in stream
            test_batch_size (int, optional): batch size for testing. Defaults to 128.
            shrinkage_param (_type_, optional): value of the shrinkage parameter for inversing the covariance matrix. Defaults to 1e-4.
            streaming_update_sigma (bool, optional): True if sigma is plastic else False
                feature extraction in `self.feature_extraction_wrapper`. Defaults to True.
            alpha (float): Regularization parameter. Defaults to 0.5.
            device (torch.device): Torch device used by the module. Defaults to torch.device('cuda').
        """
        super(StreamingRDA, self).__init__()

        self.device=device
        self.num_classes=num_classes

        # SQDA parameters
        self.input_size = input_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # SRDA weights
        self.muK = torch.zeros((num_classes, input_size)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.SigmaK = torch.ones((num_classes,input_size, input_size),dtype=torch.double).to(self.device)
        self.Sigma = torch.ones((input_size,input_size)).to(self.device)
        self.RegSigmaK = torch.empty((num_classes,input_size, input_size)).to(self.device)

        self.num_updates = torch.zeros(num_classes).to(self.device)
        self.Lambda = torch.zeros_like(self.SigmaK).to(self.device)
        self.prev_num_updates = (torch.zeros(num_classes)-1).to(self.device)
        
        self.priorK = torch.ones(num_classes).to(self.device)
        self.alpha=alpha
    
    @torch.no_grad()
    def fit(self, x, y):
        """Fit the SRDA model to a new sample (x,y).

        Args:
            x (torch.tensor): input data
            y (torch.tensor): input labels
        """

        x = x.to(self.device)
        y = y.long().to(self.device)
        
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)
        
        if self.streaming_update_sigma:
            #QDA cov update
            x_minus_mu = x - self.muK[y]
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates[y] / (self.num_updates[y] + 1)
            self.SigmaK[y] = (self.num_updates[y] * self.SigmaK[y] + delta) / (
                self.num_updates[y] + 1
            )
            #LDA cov update
            sum_updates=torch.sum(self.num_updates)
            delta2 = mult * sum_updates / (sum_updates + 1)
            self.Sigma = (sum_updates * self.Sigma + delta2) / (
                sum_updates + 1
            )

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates[y] += 1
        

    @torch.no_grad()
    def predict(self, X,return_probas=False):
        """Make predictions on test data X.

        Args:
            X (torch.tensor): tensor that contains N data samples (N x d)
            return_probas (bool): True if the user would like probabilities instead
        of predictions returned. Defaults to False.

        Returns:
            torch.tensor: returns proba or scores
        """
        X=X.to(self.device)
        #Compute priors and the regularized cov matrix
        s=torch.sum(self.cK).item()
        for i in range(self.num_classes):
            #Reg cov
            self.RegSigmaK[i]=self.alpha*self.SigmaK[i]+(1-self.alpha)*self.Sigma
            #Priors
            p=self.cK[i]/s
            if p==0:
                self.priorK[i]=1
            else:
                self.priorK[i]=p
        
        # Compute lambda if class was updated
        for i in range(self.num_classes):
            
            if self.prev_num_updates[i] != self.num_updates[i]:
                self.Lambda[i] = torch.pinverse(
                    (1 - self.shrinkage_param) * self.RegSigmaK[i]
                    + self.shrinkage_param
                    * torch.eye(self.input_size, device=self.device)
                )
                self.prev_num_updates[i] = self.num_updates[i]
        
        scores=[]

        for i in range(X.shape[0]):
            sample_score=[]
            for k in range(self.num_classes):
                if self.cK[k]==0:
                    sample_score.append(float('-inf'))
                else:
                    sample_score.append(log(self.priorK[k])-0.5*torch.log(torch.norm(self.RegSigmaK[k]))-0.5*torch.matmul((X[i]-self.muK[k]),torch.matmul(self.Lambda[k].float(),(X[i]-self.muK[k]))))
            scores.append(torch.Tensor(sample_score))
        
        scores=torch.stack(scores)
        
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores,dim=1).cpu()

    @torch.no_grad()
    def fit_base(self, X, y,init='default'):
        """Fit the SRDA model to the base data.

        Args:
            X (torch.tensor): an Nxd torch tensor of base initialization data.
            y (torch.tensor): an Nx1-dimensional torch tensor of the associated labels for X.
            init (str): Initialization Scheme. Options: ['default','OAS']. Defaults to 'default'.
        """

        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()

        if init=='default':
            for x, label in zip(X, y):
                self.fit(x, label.view(1, ))

        elif init=='OAS':
            # update class means
            for k in torch.unique(y):
                self.muK[k] = X[y == k].mean(0)
                self.cK[k] = X[y == k].shape[0]
            self.num_updates=self.cK

            print('\nEstimating initial covariance matrices...')
            
            for k in torch.unique(y):
                cov_estimator = OAS(assume_centered=True)
                cov_estimator.fit((X[y==k] - self.muK[k]).cpu().numpy())
                self.SigmaK[k] = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

            cov_estimator2 = OAS(assume_centered=True)
            cov_estimator2.fit((X - self.muK[y]).cpu().numpy())
            self.Sigma = torch.from_numpy(cov_estimator2.covariance_).float().to(self.device)
        else:
            raise NotImplementedError('Please implement another initialization scheme.')

    def save_model(self, save_path, save_name):
        """Save the model parameters to a torch file.

        Args:
            save_path (str): the path where the model will be saved
            save_name (str): the name for the saved file
        """

        #parameters
        d = dict()
        d["muK"] = self.muK.cpu()
        d["cK"] = self.cK.cpu()
        d["SigmaK"] = self.SigmaK.cpu()
        d["Sigma"] = self.Sigma.cpu()
        d["num_updates"] = self.num_updates.cpu()

        torch.save(d, os.path.join(save_path, save_name + ".pth"))

    def load_model(self, save_path, save_name):
        """Load the model parameters into StreamingRDA object.

        Args:
            save_path (str): the path where the model is saved
            save_name (str): the name of the saved file
        """

        d = torch.load(os.path.join(save_path, save_name + ".pth"))
        self.muK = d["muK"].to(self.device)
        self.cK = d["cK"].to(self.device)
        self.SigmaK = d["SigmaK"].to(self.device)
        self.Sigma = d["Sigma"].to(self.device)
        self.num_updates = d["num_updates"].to(self.device)
    
