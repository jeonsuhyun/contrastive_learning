
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ljcmp.models.latent_model import LatentValidityModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValidityNet(torch.nn.Module, LatentValidityModel):
    
    def __init__(self, h_dim, z_dim, c_dim=0, activation=F.relu):
        """validitynet

        Args:
            h_dim (int): dimension of hidden unit
            z_dim (int): dimension of latent representation
            activation (optional): _description_. Defaults to F.relu.
        """
        super(ValidityNet, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.given_c = torch.zeros((1, c_dim)).to(device=device)  
        self.activation = activation
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        if type(h_dim) is list:
            for i in range(len(h_dim)):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim, h_dim[i]).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim[i-1]+c_dim, h_dim[i]).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim[i]).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim[-1]+c_dim, 1).to(device=device)
            
        elif type(h_dim) is int:
            for i in range(3):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim, h_dim).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim+c_dim, h_dim).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim+c_dim, 1).to(device=device)

        else:
            raise ValueError("h_dim must be int or list of int")
            
        self.threshold = 0.5
        
    def forward(self, z, c=None):
        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)

        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(z)))
                z = dr(z)
            z = self.fc_logits(z)
            
        z = torch.sigmoid(z)

        return z

    def predict(self, z, c=None):
        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)

        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(z)))
                z = dr(z)
            z = self.fc_logits(z)
            
        z = torch.sigmoid(z)

        return z

    def is_valid_estimated(self, z, c=None):
        z = torch.from_numpy(z).float().to(device=device)
        c = torch.from_numpy(c).float().to(device=device) if self.c_dim > 0 and c is not None else None

        res = self.forward(z,c)
        res = res.detach().cpu().numpy()
        
        return res > self.threshold

    def set_condition(self, c):
        if self.c_dim == 0:
            print("Warning: no condition is needed for this model")
            return
        self.given_c = torch.from_numpy(c).to(device=device, dtype=torch.float32)

# class DepthValidityNet(torch.nn.Module, LatentValidityModel):

#     def __init__(self, h_dim, z_dim, c_dim=0, n_grid=32, voxel_latent_dim=16, activation=F.relu):
#         """validitynet

#         Args:
#             h_dim (int): dimension of hidden unit
#             z_dim (int): dimension of latent representation
#             activation (optional): _description_. Defaults to F.relu.
#         """
#         super(DepthValidityNet, self).__init__()

#         self.h_dim = h_dim
#         self.z_dim = z_dim
#         self.c_dim = c_dim
#         self.given_c = torch.zeros((1, c_dim)).to(device=device)  
#         self.given_voxel = torch.zeros((1, n_grid**3)).to(device=device)
#         self.given_voxel_latent = torch.zeros((1, voxel_latent_dim)).to(device=device)
#         self.n_grid = n_grid
#         self.n_grid_3 = n_grid**3
#         self.voxel_latent_dim = voxel_latent_dim
#         self.activation = activation
        
#         self.fc_layers = nn.ModuleList()
#         self.batch_norm_layers = nn.ModuleList()
#         self.dropout_layers = nn.ModuleList()

#         self.depth_cnn = nn.Sequential(
#             nn.Conv2d(1, 32, 3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Dropout(0.2),
#             nn.Conv2d(32, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Dropout(0.2),
#             nn.Conv2d(64, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())


#         # self.voxel_decoder = nn.Sequential(
#         #     nn.Linear(self.voxel_latent_dim, 256),
#         #     nn.BatchNorm1d(256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, 256),
#         #     nn.BatchNorm1d(256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, self.n_grid_3),
#         #     nn.Sigmoid(),
#         # )

#         if type(h_dim) is list:
#             for i in range(len(h_dim)):
#                 if i == 0:
#                     self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
#                 else:
#                     self.fc_layers.append(nn.Linear(h_dim[i-1]+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
#                 self.batch_norm_layers.append(nn.BatchNorm1d(h_dim[i]).to(device=device))
#                 self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
#             self.fc_logits = nn.Linear(h_dim[-1]+c_dim+self.voxel_latent_dim, 1).to(device=device)
            
#         elif type(h_dim) is int:
#             for i in range(3):
#                 if i == 0:
#                     self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
#                 else:
#                     self.fc_layers.append(nn.Linear(h_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
#                 self.batch_norm_layers.append(nn.BatchNorm1d(h_dim).to(device=device))
#                 self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
#             self.fc_logits = nn.Linear(h_dim+c_dim+self.voxel_latent_dim, 1).to(device=device)

#         else:
#             raise ValueError("h_dim must be int or list of int")
            
#         self.threshold = 0.5
    
#     def loss(self, z, c, y, y_hat):
#         # print(torch.max(y_hat))
#         # print(torch.min(y_hat))
#         # print(torch.max(voxel_hat))
#         # print(torch.min(voxel_hat))
#         # import pdb; pdb.set_trace()
#         # loss_voxel_recon = F.binary_cross_entropy(voxel, voxel_hat) / self.n_grid_3
#         loss_estimation = F.binary_cross_entropy(y_hat, y)

#         # return loss_estimation, loss_voxel_recon
#         return loss_estimation

#     def forward(self, z, voxel=None, voxel_latent=None, c=None):
#         # import pdb; pdb.set_trace()
#         # voxel_latent = self.voxel_encoder(voxel)
#         if voxel_latent is None:
#             if voxel is None:
#                 voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
#             else:
#                 voxel_latent = self.voxel_encoder(voxel)
#         # else:
#             # voxel_latent = voxel_latent.repeat(z.shape[0], 1)
#         # voxel_hat = self.voxel_decoder(voxel_latent)

#         if self.c_dim > 0 and c is None:
#             c = self.given_c.repeat(z.shape[0], 1)

#         if self.c_dim > 0:
#             for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
#                 z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
#                 z = dr(z)
#             z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
#         else:
#             for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
#                 z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
#                 z = dr(z)
#             z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))
            
#         y_hat = torch.sigmoid(z)

#         return y_hat, voxel_latent
#         # return y_hat, voxel_latent, voxel_hat

#     def predict(self, z, c=None, voxel=None):
#         if voxel is None:
#             voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
#         else:
#             voxel_latent = self.voxel_encoder(voxel)

#         if self.c_dim > 0 and c is None:
#             c = self.given_c.repeat(z.shape[0], 1)
        
#         if self.c_dim > 0:
#             for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
#                 z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
#                 z = dr(z)
#             z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
#         else:
#             for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
#                 z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
#                 z = dr(z)
#             z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))

#         y_hat = torch.sigmoid(z)
        
#         return y_hat


#     def is_valid_estimated(self, z, c=None):
#         z = torch.from_numpy(z).float().to(device=device)
#         c = torch.from_numpy(c).float().to(device=device) if self.c_dim > 0 and c is not None else None

#         res = self.predict(z,c)
#         res = res.detach().cpu().numpy()
        
#         return res > self.threshold

#     def set_condition(self, c):
#         if self.c_dim == 0:
#             print("Warning: no condition is needed for this model")
#             return
#         self.given_c = torch.from_numpy(c).to(device=device, dtype=torch.float32)

#     def set_voxel(self, voxel):
#         if type(voxel) is np.ndarray:
#             # voxel = voxel.reshape(1,-1) # flatten
#             self.voxel = torch.from_numpy(voxel).to(device=device, dtype=torch.float32)
#         elif type(voxel) is torch.Tensor:
#             self.voxel = voxel.to(device=device, dtype=torch.float32)
#         self.voxel = self.voxel[None, :]
#         self.given_voxel_latent = self.voxel_encoder(self.voxel)

class VoxelValidityNet(torch.nn.Module, LatentValidityModel):
    
    def __init__(self, h_dim, z_dim, c_dim=0, n_grid=32, voxel_latent_dim=16, activation=F.relu):
        """validitynet

        Args:
            h_dim (int): dimension of hidden unit
            z_dim (int): dimension of latent representation
            activation (optional): _description_. Defaults to F.relu.
        """
        super(VoxelValidityNet, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.given_c = torch.zeros((1, c_dim)).to(device=device)  
        self.given_voxel = torch.zeros((1, n_grid**3)).to(device=device)
        self.given_voxel_latent = torch.zeros((1, voxel_latent_dim)).to(device=device)
        self.n_grid = n_grid
        self.n_grid_3 = n_grid**3
        self.voxel_latent_dim = voxel_latent_dim
        self.activation = activation
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.voxel_encoder = nn.Sequential(
            nn.Linear(self.n_grid_3, 512),
            nn.BatchNorm1d(512),
            nn.Dropout1d(0.2),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout1d(0.2),
            nn.LeakyReLU(),
            nn.Linear(512, self.voxel_latent_dim)
        )

        # self.voxel_decoder = nn.Sequential(
        #     nn.Linear(self.voxel_latent_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_grid_3),
        #     nn.Sigmoid(),
        # )

        if type(h_dim) is list:
            for i in range(len(h_dim)):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim[i-1]+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim[i]).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim[-1]+c_dim+self.voxel_latent_dim, 1).to(device=device)
            
        elif type(h_dim) is int:
            for i in range(3):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim+c_dim+self.voxel_latent_dim, 1).to(device=device)

        else:
            raise ValueError("h_dim must be int or list of int")
            
        self.threshold = 0.5
    
    def loss(self, z, c, y, y_hat):
        # print(torch.max(y_hat))
        # print(torch.min(y_hat))
        # print(torch.max(voxel_hat))
        # print(torch.min(voxel_hat))
        # import pdb; pdb.set_trace()
        # loss_voxel_recon = F.binary_cross_entropy(voxel, voxel_hat) / self.n_grid_3
        loss_estimation = F.binary_cross_entropy(y_hat, y)

        # return loss_estimation, loss_voxel_recon
        return loss_estimation

    def forward(self, z, voxel=None, voxel_latent=None, c=None):
        # import pdb; pdb.set_trace()
        # voxel_latent = self.voxel_encoder(voxel)
        if voxel_latent is None:
            if voxel is None:
                voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
            else:
                voxel_latent = self.voxel_encoder(voxel)
        # else:
            # voxel_latent = voxel_latent.repeat(z.shape[0], 1)
        # voxel_hat = self.voxel_decoder(voxel_latent)

        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)

        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))
            
        y_hat = torch.sigmoid(z)

        return y_hat, voxel_latent
        # return y_hat, voxel_latent, voxel_hat

    def predict(self, z, c=None, voxel=None):
        if voxel is None:
            voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
        else:
            voxel_latent = self.voxel_encoder(voxel)

        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)
        
        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))

        y_hat = torch.sigmoid(z)
        
        return y_hat


    def is_valid_estimated(self, z, c=None):
        z = torch.from_numpy(z).float().to(device=device)
        c = torch.from_numpy(c).float().to(device=device) if self.c_dim > 0 and c is not None else None

        res = self.predict(z,c)
        res = res.detach().cpu().numpy()
        
        return res > self.threshold

    def set_condition(self, c):
        if self.c_dim == 0:
            print("Warning: no condition is needed for this model")
            return
        self.given_c = torch.from_numpy(c).to(device=device, dtype=torch.float32)

    def set_voxel(self, voxel):
        if type(voxel) is np.ndarray:
            # voxel = voxel.reshape(1,-1) # flatten
            self.voxel = torch.from_numpy(voxel).to(device=device, dtype=torch.float32)
        elif type(voxel) is torch.Tensor:
            self.voxel = voxel.to(device=device, dtype=torch.float32)
        self.voxel = self.voxel[None, :]
        self.given_voxel_latent = self.voxel_encoder(self.voxel)


class VoxelCNNValidityNet(torch.nn.Module, LatentValidityModel):
    
    def __init__(self, h_dim, z_dim, c_dim=0, n_grid=32, voxel_latent_dim=16, activation=F.relu):
        """validitynet

        Args:
            h_dim (int): dimension of hidden unit
            z_dim (int): dimension of latent representation
            activation (optional): _description_. Defaults to F.relu.
        """
        super(VoxelCNNValidityNet, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.given_c = torch.zeros((1, c_dim)).to(device=device)  
        self.given_voxel = torch.zeros((1, n_grid**3)).to(device=device)
        self.given_voxel_latent = torch.zeros((1, voxel_latent_dim)).to(device=device)
        self.n_grid = n_grid
        self.n_grid_3 = n_grid**3
        self.voxel_latent_dim = voxel_latent_dim
        self.activation = activation
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        self.voxel_cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU())

        # self.voxel_decoder = nn.Sequential(
        #     nn.Linear(self.voxel_latent_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_grid_3),
        #     nn.Sigmoid(),
        # )

        if type(h_dim) is list:
            for i in range(len(h_dim)):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim[i-1]+c_dim+self.voxel_latent_dim, h_dim[i]).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim[i]).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim[-1]+c_dim+self.voxel_latent_dim, 1).to(device=device)
            
        elif type(h_dim) is int:
            for i in range(3):
                if i == 0:
                    self.fc_layers.append(nn.Linear(z_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
                else:
                    self.fc_layers.append(nn.Linear(h_dim+c_dim+self.voxel_latent_dim, h_dim).to(device=device))
                self.batch_norm_layers.append(nn.BatchNorm1d(h_dim).to(device=device))
                self.dropout_layers.append(nn.Dropout(0.2).to(device=device))
            self.fc_logits = nn.Linear(h_dim+c_dim+self.voxel_latent_dim, 1).to(device=device)

        else:
            raise ValueError("h_dim must be int or list of int")
            
        self.threshold = 0.5
    
    def loss(self, z, c, y, y_hat):
        # print(torch.max(y_hat))
        # print(torch.min(y_hat))
        # print(torch.max(voxel_hat))
        # print(torch.min(voxel_hat))
        # import pdb; pdb.set_trace()
        # loss_voxel_recon = F.binary_cross_entropy(voxel, voxel_hat) / self.n_grid_3
        loss_estimation = F.binary_cross_entropy(y_hat, y)

        # return loss_estimation, loss_voxel_recon
        return loss_estimation

    def forward(self, z, voxel=None, voxel_latent=None, c=None):
        # import pdb; pdb.set_trace()
        # voxel_latent = self.voxel_encoder(voxel)
        if voxel_latent is None:
            if voxel is None:
                voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
            else:
                voxel_latent = self.voxel_encoder(voxel)
        # else:
            # voxel_latent = voxel_latent.repeat(z.shape[0], 1)
        # voxel_hat = self.voxel_decoder(voxel_latent)

        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)

        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))
            
        y_hat = torch.sigmoid(z)

        return y_hat, voxel_latent
        # return y_hat, voxel_latent, voxel_hat

    def predict(self, z, c=None, voxel=None):
        if voxel is None:
            voxel_latent = self.given_voxel_latent.repeat(z.shape[0], 1)
        else:
            voxel_latent = self.voxel_encoder(voxel)

        if self.c_dim > 0 and c is None:
            c = self.given_c.repeat(z.shape[0], 1)
        
        if self.c_dim > 0:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,c,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,c,voxel_latent),axis=1))
        else:
            for fc, bn, dr in zip(self.fc_layers, self.batch_norm_layers, self.dropout_layers):
                z = self.activation(bn(fc(torch.cat((z,voxel_latent),axis=1))))
                z = dr(z)
            z = self.fc_logits(torch.cat((z,voxel_latent),axis=1))

        y_hat = torch.sigmoid(z)
        
        return y_hat


    def is_valid_estimated(self, z, c=None):
        z = torch.from_numpy(z).float().to(device=device)
        c = torch.from_numpy(c).float().to(device=device) if self.c_dim > 0 and c is not None else None

        res = self.predict(z,c)
        res = res.detach().cpu().numpy()
        
        return res > self.threshold

    def set_condition(self, c):
        if self.c_dim == 0:
            print("Warning: no condition is needed for this model")
            return
        self.given_c = torch.from_numpy(c).to(device=device, dtype=torch.float32)

    def set_voxel(self, voxel):
        if type(voxel) is np.ndarray:
            # voxel = voxel.reshape(1,-1) # flatten
            self.voxel = torch.from_numpy(voxel).to(device=device, dtype=torch.float32)
        elif type(voxel) is torch.Tensor:
            self.voxel = voxel.to(device=device, dtype=torch.float32)
        self.voxel = self.voxel[None, :]
        self.given_voxel_latent = self.voxel_encoder(self.voxel)