from dolfin import *
from dolfin_adjoint import *


def neohookean(I_1, I_2, J, E=100, nu=0.3, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    psi = mu / 2 * (I_1 - 3 - 2*ln(J)) + lamda / 2 * (J - 1) ** 2
    return psi


def mooney_rivlin2(I_1, I_2, J, E=100, nu=0.3, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2*mu/3
    # normalize
    I_1_bar = J**(-2/3)*I_1
    I_2_bar = J**(-4/3)*I_2
    C_01 = mu/4
    C_10 = mu/2 - C_01
    psi = C_10*(I_1_bar - 3) + C_01*(I_2_bar - 3) + kappa/2*(J-1)**2
    return psi


def mooney_rivlin1(I_1, I_2, J, E=100, nu=0.3, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2*mu/3
    # normalize
    I_2_bar = J**(-4/3)*I_2
    C_01 = mu/2
    psi = C_01*(I_2_bar - 3) + kappa/2*(J-1)**2
    return psi


def mooney_rivlin3(I_1, I_2, J, E=100, nu=0.3, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2*mu/3
    # normalize
    I_2_bar = J**(-4/3)*I_2
    C_01 = mu/2
    psi = C_01*(I_2_bar - 3) + kappa/2*( (J**2-1)/2-ln(J) )
    return psi


def gent(I_1, I_2, J, E=100, nu=0.3, Jm=0.2, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2 * mu / 3
    I_1_bar = J ** (-2 / 3) * I_1
    psi = -Jm * mu * ln(1 - (I_1_bar - 3) / Jm) / 2 + kappa/2*( (J**2-1)/2-ln(J) )
    return psi


def gent_new(I_1, I_2, J, E=100, nu=0.3, Jm=0.4, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2 * mu / 3
    psi = -Jm*mu*ln(1 - (I_1-3)/Jm)/2 + mu*(J-1)*(J-3)/2 - mu*(J-1)**2/Jm + lamda * (J-1)**2/2
    return psi


def gent_donald(I_1, I_2, J, E=100, nu=0.3, Jm=0.1, **kwargs):
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    kappa = lamda + 2 * mu / 3
    psi = -Jm*mu*ln(1 - (I_1-3)/Jm)/2 + mu*(J-1)*(J-3)/2 + lamda * (J-1)**2/2
    return psi


def linear_elastic(theta, epsilon, epsilon33, E=100, nu=0.3, **kwargs):
    '''
    :param theta: 3D
    :param epsilon: 2D
    :param epsilon33:
    :param E:
    :param nu:
    :param kwargs:
    :return:
    '''
    lamda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    psi = 1/2*(lamda*theta**2 + 2*mu*(inner(epsilon, epsilon) + epsilon33**2))
    return psi

def get_constitutive(consti_name):
    assert consti_name in ["NH", "MR2", "MR1", "MR3", "Gent", "Gent_New", "Gent_Donald", None]
    if consti_name == "NH":
        return neohookean
    elif consti_name == "MR2":
        return mooney_rivlin2
    elif consti_name == "MR1":
        return mooney_rivlin1
    elif consti_name == "MR3":
        return mooney_rivlin3
    elif consti_name == "Gent":
        return gent
    elif consti_name == "Gent_New":
        return gent_new
    elif consti_name == "Gent_Donald":
        return gent_donald
    elif consti_name == None:
        return None
