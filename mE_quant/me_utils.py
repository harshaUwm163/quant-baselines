import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def fp_bias(e_bit):
    return (1 << (e_bit - 1)) - 1

def fp_nmax(e_bit, m_bit, support_special_values = False):
    bias = fp_bias(e_bit)
    max_exp = (1 << e_bit) - 1 - bias
    if support_special_values:
        max_exp -= 1

    max_frac = ((1 << (m_bit + 1)) - 1) / (1 << m_bit)
    return max_frac * (2**max_exp)

def fp_nmin(e_bit):
    bias = fp_bias(e_bit)
    min_exp = 1 - bias
    return 1.0 * (2**min_exp)

def fp_round(   x,
                scale,
                mu,
                e_bit,
                m_bit,
                support_special_values = False):
    x -= mu
    x /= scale
    nmax = fp_nmax(e_bit, m_bit, support_special_values)
    nmin = fp_nmin(e_bit)
    # saturate when overflow
    x_overflow = torch.sign(x) * nmax
    # sub-normal login when underflow
    fraction_size = 1 << m_bit
    x_underflow_scaled = torch.round(x / nmin * fraction_size)
    x_underflow = x_underflow_scaled * nmin / fraction_size
    # Round data with full exponent range 
    _, x_exp = torch.frexp(x)
    x_normal = torch.round(x * 2**(-x_exp)) * 2**(x_exp)
    x = torch.where(torch.abs(x) > nmax, x_overflow, x_normal)
    x = torch.where(torch.abs(x) < nmin, x_underflow, x)

    return x * scale + mu

def fp_round_error( x,
                    scale, 
                    mu, 
                    e_bit, 
                    m_bit, 
                    support_special_values = False,
                    ridge_lambda = 0.0,
                    barrier_nu = 0.0):
    # relying on pytorch broadcasting
    y = (x - mu) / scale
    nmax = fp_nmax(e_bit, m_bit, support_special_values) 
    nmin = fp_nmin(e_bit) 
    eps = 2**(-m_bit - 1)

    # overflow error
    Overflow_err = torch.abs(x - mu) - (1-eps) * nmax * scale
    ridge_error = ridge_lambda * (y/nmax + 1) * (y/nmax - 1)
    underflow_err = eps * scale + ridge_error

    err = torch.abs(x - mu) * eps + ridge_error
    err = torch.where(torch.abs(y) < nmin, underflow_err, err)
    err = torch.where(torch.abs(y) > nmax, Overflow_err, err)

    return err - barrier_nu * eps/nmax * torch.log(scale)

def fi_max(m_bit):
    return (1 << (m_bit - 1)) - 1

def fi_min(m_bit):
    return -(1 << (m_bit - 1))

def fi_round(x, 
             scale,
             mu,
             m_bit,
             ):
    x = (x - mu) / scale
    x_fi_max = fi_max(m_bit)
    x_fi_min = fi_min(m_bit)
    x = torch.round(x)
    x = torch.where(x > x_fi_max, x_fi_max, x)
    x = torch.where(x < x_fi_min, x_fi_min, x)

    return x * scale + mu

# Ensure we can handle asymmetric nature of fixed point container
# i.e., int2 is [-2,1], int4 is [-8,7], int8 is [-128,127]
# plot the error corresponding to scale, sanity check for convexity.
def fi_round_error(x, 
                   scale, 
                   mu, 
                   m_bit, 
                   ridge_lambda = 0.0, 
                   barrier_nu = 0.0):
    # relying on the pytorch broadcasting
    y = (x - mu) / scale
    y_fi_max = fi_max(m_bit)
    y_fi_min = fi_min(m_bit)

    y_mu = (y_fi_max + y_fi_min) / 2
    y_diff = (y_fi_max - y_fi_min) / 2

    underflow_err = 0.5 + ridge_lambda * ((y - y_mu)/y_diff + 1) * ((y-y_mu)/y_diff - 1)
    pos_overflow_err = y - y_fi_max + 0.5
    neg_overflow_err = -y + y_fi_min + 0.5
    err = torch.where(y > y_fi_max, pos_overflow_err, underflow_err)
    err = torch.where(y < y_fi_min, neg_overflow_err, err)

    return err * scale - barrier_nu * 0.5 / y_fi_max * torch.log(scale)
    