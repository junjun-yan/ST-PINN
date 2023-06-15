import numpy as np
import tensorflow as tf


def Burgers1D(x, t, u, nu):
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]

    e = u_t + u * u_x - nu / np.pi * u_xx
    return e


def diffusion_reaction_1d(x, t, u, nu, rho):
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    e = u_t - nu * u_xx - rho * (u - tf.multiply(u, u))
    return e


def Diffusion_sorption(x, t, u):
    D: float = 5e-4
    por: float = 0.29
    rho_s: float = 2880
    k_f: float = 3.5e-4
    n_f: float = 0.874
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u + 1e-6) ** (n_f - 1)

    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, t)[0]

    return u_t - D / retardation_factor * u_xx


def Boundary(x, u):
    D: float = 5e-4
    u_x = tf.gradients(u, x)[0]

    return D * u_x


def SWE_2D(u, v, h, x, y, t, g):
    u_t = tf.gradients(u, t)[0]
    v_t = tf.gradients(v, t)[0]
    h_t = tf.gradients(h, t)[0]

    u_x = tf.gradients(u, x)[0]
    v_x = tf.gradients(v, x)[0]
    h_x = tf.gradients(h, x)[0]

    u_y = tf.gradients(u, y)[0]
    v_y = tf.gradients(v, y)[0]
    h_y = tf.gradients(h, y)[0]

    e1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    e2 = u_t + u * u_x + v * u_y + g * h_x
    e3 = v_t + u * v_x + v * v_y + g * h_y

    return e1, e2, e3


def Boundary_condition(x, y, u, v):
    u_x = tf.gradients(u, x)[0]
    v_x = tf.gradients(v, x)[0]

    u_y = tf.gradients(u, y)[0]
    v_y = tf.gradients(v, y)[0]

    return u_x, v_x, u_y, v_y


def CFD_2D(x, y, t, d, u, v, p, gamma, keci, yifu):
    E = p / (gamma - 1.) + 0.5 * d * (u ** 2 + v ** 2)
    Fu = u * (E + p)
    Fv = v * (E + p)
    du = d * u
    dv = d * v

    d_t = tf.gradients(d, t)[0]
    du_x = tf.gradients(du, x)[0]
    dv_y = tf.gradients(dv, y)[0]
    u_t = tf.gradients(u, t)[0]
    u_x = tf.gradients(u, x)[0]
    u_y = tf.gradients(u, y)[0]
    v_t = tf.gradients(v, t)[0]
    v_x = tf.gradients(v, x)[0]
    v_y = tf.gradients(v, y)[0]
    p_x = tf.gradients(p, x)[0]
    p_y = tf.gradients(p, y)[0]
    E_t = tf.gradients(E, t)[0]
    Fu_x = tf.gradients(Fu, x)[0]
    Fv_y = tf.gradients(Fv, y)[0]
    u_xx = tf.gradients(u_x, x)[0]
    v_xx = tf.gradients(v_x, x)[0]
    u_yy = tf.gradients(u_y, y)[0]
    v_yy = tf.gradients(v_y, y)[0]
    v_yx = tf.gradients(v_y, x)[0]
    u_xy = tf.gradients(u_x, y)[0]

    e1 = d_t + du_x + dv_y
    e2 = d * (u_t + u * u_x + v * u_y) + p_x - keci * (u_xx + u_yy) - (keci + yifu / 3.0) * (u_xx + v_yx)
    e3 = d * (v_t + u * v_x + v * v_y) + p_y - keci * (v_xx + v_yy) - (keci + yifu / 3.0) * (u_xy + v_yy)
    #     e2 = d * (u_t + u * u_x + v * u_y) + p_x
    #     e3 = d * (v_t + u * v_x + v * v_y) + p_y
    e4 = E_t + Fu_x + Fv_y

    return e1, e2, e3, e4
