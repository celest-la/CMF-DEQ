# define the CMF operator and its adjoint

from numpy import newaxis, sqrt, exp, reshape
import torch as torch
import numpy as np
import deepinv as dinv
class ProbeParam:
    def __init__(self, device='cpu'):
        self.name = 'L7-4'
        self.fs = 22e6
        self.c0 = 1480
        self.fc = [3.5e6, 7.5e6]
        self.Ne = 128
        self.Pitch = 3e-4
        self.fcentrale = 5.5e6

        # Define xp and move it to the specified device
        xp = np.arange(0, self.Ne) * self.Pitch
        xp = xp - xp[-1] / 2
        self.xp = torch.from_numpy(xp).to(device)

        # Store the device
        self.device = device

def get_steering_vectors(X, Z, h, frecon):
    # Flatten and reshape X and Z
    flatx = X.flatten().unsqueeze(1)  # Add new axis
    xp = h.xp.unsqueeze(0)  # Reshape xp for broadcasting
    distx = flatx - xp  # Compute distance in x-axis

    # Compute distances along z-axis
    flatz = Z.flatten().unsqueeze(1)
    distz = flatz.repeat(1, len(h.xp))  # Equivalent to np.tile

    # Total distance
    dist = torch.sqrt(distx ** 2 + distz ** 2)
    dist_center = torch.sqrt(flatx ** 2 + distz ** 2)
    tau_pix = (dist - dist_center) / h.c0  # Time delay

    # Steering vector computation
    steer_vec = torch.exp(-2j * torch.pi * frecon * tau_pix)
    steer_vec = steer_vec.T  # Transpose for desired shape

    # Normalize the steering vectors
    steer_vec = torch.nn.functional.normalize(steer_vec, dim=1)
    return steer_vec.to(torch.complex64)


def complexToReal(X):
    #return torch.vstack((X.real, X.imag))
    real_part = X.real  # Shape: (N, C, m, m)
    imag_part = X.imag  # Shape: (N, C, m, m)

    # Concatenate along the third dimension (dim=2)
    return torch.cat((real_part, imag_part), dim=2)  # Shape: (N, C, 2*m, m)

class CmfOperator(dinv.physics.LinearPhysics):
    r"""
    Create the CSM from the power map. The resulting power map is a complex square matrix.

    The adjoint operation is equivalent to the DAS beamforming.

    """

    def __init__(
          self,
          frecon,
          h,
          X,
          Z,
          device="cpu",
          **kwargs
  ):
      super().__init__(**kwargs)
      self.h = h
      self.col = X.shape[0]
      self.line = X.shape[1]
      self.frecon = frecon
      self.X = X
      self.Z = Z
      self.device=device
      self.steering_vectors = get_steering_vectors(X, Z, h, frecon).to(device)

    def A(self, x, **kwargs):
        """
        Forward operator: computes y = S diag(x_flat) S^H efficiently without forming large diagonal matrices.
        """
        N, C = x.shape[:2]
        P = x.numel() // (N * C)
        x_flat = x.view(N, C, P)          # shape (N, C, J)
        S = self.steering_vectors        # shape (I, J)
        S_conj = torch.conj(S)           # shape (I, J)

        # y_complex[a, b, i, k] = sum_j S[i, j] * x_flat[a, b, j] * S_conj[k, j]
        y_complex = torch.einsum('ij,abj,kj->abik', S, x_flat, S_conj)
        y = complexToReal(y_complex)
        return y.float()

    def A_adjoint(self, y, theta=None):
        """
        Adjoint operator: computes x = diag(S^H y_complex S) efficiently by directly computing diagonal entries.
        """
        m = self.h.Ne
        N, C = y.shape[:2]
        P = self.col * self.line

        # reconstruct complex ytemp of shape (N, C, m, P)
        y_real = y[:, :, :m, :]
        y_imag = y[:, :, m:, :]
        ytemp = y_real + 1j * y_imag     # shape (N, C, I, J)

        S = self.steering_vectors        # shape (I, J)
        S_conj = torch.conj(S)           # shape (I, J)

        # xtemp[a, b, j] = sum_{i,k} S_conj[i, j] * ytemp[a, b, i, k] * S[k, j]
        xtemp = torch.einsum('ij,abik,kj->abj', S_conj, ytemp, S)
        xtemp = torch.real(xtemp)        # take real part
        xtemp = torch.nn.functional.normalize(xtemp, dim=2)

        x = xtemp.view(N, C, self.col, self.line)
        return x.float()
