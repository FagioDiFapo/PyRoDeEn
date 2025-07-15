import numpy as np

def minmod(v):
    # Using Harten's generalized definition
    v = np.array(v)
    s = np.sum(np.sign(v)) / len(v)
    if abs(s) == 1:
        return s * np.min(np.abs(v))
    else:
        return 0.0

def vanalbada(da, db, h):
    # Van Albada Slope Limiter Function
    eps2 = (0.3 * h) ** 3
    numerator = (db ** 2 + eps2) * da + (da ** 2 + eps2) * db
    denominator = da ** 2 + db ** 2 + 2 * eps2
    if denominator != 0:
        return 0.5 * (np.sign(da) * np.sign(db) + 1) * numerator / denominator
    else:
        return 0.0

def vanLeer(da, db):
    # Van Leer Slope Limiter Function
    if db != 0:
        r = da / db
        return (r + abs(r)) / (1 + abs(r))
    else:
        return 0.0

def HLLE1Dflux(qL, qR, normal, gamma=1.4):
    # Compute HLLE flux for Euler equations in 2D
    nx, ny = normal

    # Left state
    rL = qL[0]
    uL = qL[1] / rL
    vL = qL[2] / rL
    vnL = uL * nx + vL * ny
    pL = (gamma - 1) * (qL[3] - rL * (uL ** 2 + vL ** 2) / 2)
    aL = np.sqrt(gamma * pL / rL)
    HL = (qL[3] + pL) / rL

    # Right state
    rR = qR[0]
    uR = qR[1] / rR
    vR = qR[2] / rR
    vnR = uR * nx + vR * ny
    pR = (gamma - 1) * (qR[3] - rR * (uR ** 2 + vR ** 2) / 2)
    aR = np.sqrt(gamma * pR / rR)
    HR = (qR[3] + pR) / rR

    # Roe averages
    RT = np.sqrt(rR / rL)
    u = (uL + RT * uR) / (1 + RT)
    v = (vL + RT * vR) / (1 + RT)
    H = (HL + RT * HR) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - (u ** 2 + v ** 2) / 2))
    vn = u * nx + v * ny

    # Wave speed estimates
    SLm = min(vnL - aL, vn - a, 0)
    SRp = max(vnR + aR, vn + a, 0)

    # Left and Right fluxes
    FL = np.array([rL * vnL,
                   rL * vnL * uL + pL * nx,
                   rL * vnL * vL + pL * ny,
                   rL * vnL * HL])
    FR = np.array([rR * vnR,
                   rR * vnR * uR + pR * nx,
                   rR * vnR * vR + pR * ny,
                   rR * vnR * HR])

    # HLLE flux
    if SRp - SLm != 0:
        HLLE = (SRp * FL - SLm * FR + SLm * SRp * (qR - qL)) / (SRp - SLm)
    else:
        HLLE = np.zeros_like(qL)
    return HLLE



def HLLE2Dflux(qSW, qSE, qNW, qNE, gamma=1.4):
    # Compute HLLE flux
    # West state
    rSW = qSW[0]
    uSW = qSW[1] / rSW
    vSW = qSW[2] / rSW
    pSW = (gamma-1) * (qSW[3] - rSW * (uSW**2 + vSW**2) / 2)
    aSW = np.sqrt(gamma * pSW / rSW)
    HSW = (qSW[3] + pSW) / rSW

    # East state
    rSE = qSE[0]
    uSE = qSE[1] / rSE
    vSE = qSE[2] / rSE
    pSE = (gamma-1) * (qSE[3] - rSE * (uSE**2 + vSE**2) / 2)
    aSE = np.sqrt(gamma * pSE / rSE)
    HSE = (qSE[3] + pSE) / rSE

    # South state
    rNW = qNW[0]
    uNW = qNW[1] / rNW
    vNW = qNW[2] / rNW
    pNW = (gamma-1) * (qNW[3] - rNW * (uNW**2 + vNW**2) / 2)
    aNW = np.sqrt(gamma * pNW / rNW)
    HNW = (qNW[3] + pNW) / rNW

    # North state
    rNE = qNE[0]
    uNE = qNE[1] / rNE
    vNE = qNE[2] / rNE
    pNE = (gamma-1) * (qNE[3] - rNE * (uNE**2 + vNE**2) / 2)
    aNE = np.sqrt(gamma * pNE / rNE)
    HNE = (qNE[3] + pNE) / rNE

    # Compute Roe Averages - SW to SE
    rSroe = np.sqrt(rSE / rSW)
    uSroe = (uSW + rSroe * uSE) / (1 + rSroe)
    vSroe = (vSW + rSroe * vSE) / (1 + rSroe)
    HSroe = (HSW + rSroe * HSE) / (1 + rSroe)
    aSroe = np.sqrt((gamma-1) * (HSroe - 0.5 * (uSroe**2 + vSroe**2)))

    # Compute Roe Averages - NW to NE
    rNroe = np.sqrt(rNE / rNW)
    uNroe = (uNW + rNroe * uNE) / (1 + rNroe)
    vNroe = (vNW + rNroe * vNE) / (1 + rNroe)
    HNroe = (HNW + rNroe * HNE) / (1 + rNroe)
    aNroe = np.sqrt((gamma-1) * (HNroe - 0.5 * (uNroe**2 + vNroe**2)))

    # Compute Roe Averages - SW to NW
    rWroe = np.sqrt(rNW / rSW)
    uWroe = (uSW + rWroe * uNW) / (1 + rWroe)
    vWroe = (vSW + rWroe * vNW) / (1 + rWroe)
    HWroe = (HSW + rWroe * HNW) / (1 + rWroe)
    aWroe = np.sqrt((gamma-1) * (HWroe - 0.5 * (uWroe**2 + vWroe**2)))

    # Compute Roe Averages - SE to NE
    rEroe = np.sqrt(rNE / rSE)
    uEroe = (uSE + rEroe * uNE) / (1 + rEroe)
    vEroe = (vSE + rEroe * vNE) / (1 + rEroe)
    HEroe = (HSE + rEroe * HNE) / (1 + rEroe)
    aEroe = np.sqrt((gamma-1) * (HEroe - 0.5 * (uEroe**2 + vEroe**2)))

    # Wave speed estimates in the S
    sSW = min([uSW - aSW, uSW + aSW, uSroe - aSroe, uSroe + aSroe])
    sSE = max([uSE - aSE, uSE + aSE, uSroe - aSroe, uSroe + aSroe])

    # Wave speed estimates in the N
    sNW = min([uNW - aNW, uNW + aNW, uNroe - aNroe, uNroe + aNroe])
    sNE = max([uNE - aNE, uNE + aNE, uNroe - aNroe, uNroe + aNroe])

    # Wave speed estimates in the W
    sWS = min([vSW - aSW, vSW + aSW, vWroe - aWroe, vWroe + aWroe])
    sWN = max([vNW - aNW, vNW + aNW, vWroe - aWroe, vWroe + aWroe])

    # Wave speed estimates in the E
    sES = min([vSE - aSE, vSE + aSE, vEroe - aEroe, vEroe + aEroe])
    sEN = max([vNE - aNE, vNE + aNE, vEroe - aEroe, vEroe + aEroe])

    # Compute fluxes
    fSW = np.array([rSW * uSW, rSW * uSW**2 + pSW, rSW * vSW * uSW, rSW * uSW * HSW])
    fSE = np.array([rSE * uSE, rSE * uSE**2 + pSE, rSE * vSE * uSE, rSE * uSE * HSE])
    fNW = np.array([rNW * uNW, rNW * uNW**2 + pNW, rNW * vNW * uNW, rNW * uNW * HNW])
    fNE = np.array([rNE * uNE, rNE * uNE**2 + pNE, rNE * vNE * uNE, rNE * uNE * HNE])

    gSW = np.array([rSW * vSW, rSW * vSW * uSW, rSW * vSW**2 + pSW, rSW * vSW * HSW])
    gSE = np.array([rSE * vSE, rSE * vSE * uSE, rSE * vSE**2 + pSE, rSE * vSE * HSE])
    gNW = np.array([rNW * vNW, rNW * vNW * uNW, rNW * vNW**2 + pNW, rNW * vNW * HNW])
    gNE = np.array([rNE * vNE, rNE * vNE * uNE, rNE * vNE**2 + pNE, rNE * vNE * HNE])

    # Compute the intermediate states
    qSO = (sSE * qSE - sSW * qSW + fSW - fSE) / (sSE - sSW)
    qNO = (sNE * qNE - sNW * qNW + fNW - fNE) / (sNE - sNW)
    qOW = (sWN * qNW - sWS * qSW + gSW - gNW) / (sWN - sWS)
    qOE = (sEN * qNE - sES * qSE + gSE - gNE) / (sEN - sES)

    # Compute the intermediate states fluxes (normal HLLE 1d fluxes)
    fSO = (sSE * fSW - sSW * fSE + sSW * sSE * (qSE - qSW)) / (sSE - sSW)
    fNO = (sNE * fNW - sNW * fNE + sNW * sNE * (qNE - qNW)) / (sNE - sNW)
    gOW = (sWN * gSW - sWS * gNW + sWS * sWN * (qNW - qSW)) / (sWN - sWS)
    gOE = (sEN * gSE - sES * gNE + sES * sEN * (qNE - qSE)) / (sEN - sES)

    # Compute the transverse intermediate fluxes (Balsara's solution)
    fOW = np.array([qOW[1], gOW[2] + (qOW[1]**2 - qOW[2]**2) / qOW[0], qOW[2] * qOW[1] / qOW[0], qOW[1] * gOW[3] / qOW[2]])
    fOE = np.array([qOE[1], gOE[2] + (qOE[1]**2 - qOE[2]**2) / qOE[0], qOE[2] * qOE[1] / qOE[0], qOE[1] * gOE[3] / qOE[2]])
    gSO = np.array([qSO[2], qSO[1] * qSO[2] / qSO[0], fSO[1] + (qSO[2]**2 - qSO[1]**2) / qSO[0], qSO[2] * fSO[3] / qSO[1]])
    gNO = np.array([qNO[2], qNO[1] * qNO[2] / qNO[0], fNO[1] + (qNO[2]**2 - qNO[1]**2) / qNO[0], qNO[2] * fNO[3] / qNO[1]])


    # Strongly Interacting state q**
    qOO = 1 / ((sNE - sSW) * (sWN - sES) + (sEN - sWS) * (sSE - sNW)) * (
         (sWN * sNE + sSE * sEN) * qNE - (sEN * sNW + sSW * sWN) * qNW +
         (sES * sSW + sNW * sWN) * qSW - (sWS * sSE + sNE * sES) * qSE
       - sWN * fNE + sEN * fNW - sES * fSW + sWS * fSE - (sEN - sES) * fOE + (sWN - sWS) * fOW
       - sSE * gNE + sSW * gNW - sNW * gSW + sNE * gSE - (sNE - sNW) * gNO + (sSE - sSW) * gSO )

    # Compute fluxes of the strongly interacting state:
    # Precompute deltas
    dq1 = sNW * sEN - sWN * sNE
    df1 = sWN - sEN
    dg1 = sNE - sNW
    dq2 = sSW * sWN - sWS * sNW
    df2 = sWS - sWN
    dg2 = sNW - sSW
    dq3 = sSE * sWS - sES * sSW
    df3 = sES - sWS
    dg3 = sSW - sSE
    dq4 = sNE * sES - sEN * sSE
    df4 = sEN - sES
    dg4 = sSE - sNE

    # Using LSQ
    b1 = dq1 * (qNO - qOO) + df1 * fNO + dg1 * gNO
    b2 = dq2 * (qOW - qOO) + df2 * fOW + dg2 * gOW
    b3 = dq3 * (qSO - qOO) + df3 * fSO + dg3 * gSO
    b4 = dq4 * (qOE - qOO) + df4 * fOE + dg4 * gOE

    # k-weights
    k11 = df1 * (dg2**2 + dg3**2 + dg4**2) - dg1 * (df2 * dg2 + df3 * dg3 + df4 * dg4)
    k12 = df2 * (dg1**2 + dg3**2 + dg4**2) - dg2 * (df1 * dg1 + df3 * dg3 + df4 * dg4)
    k13 = df3 * (dg1**2 + dg2**2 + dg4**2) - dg3 * (df1 * dg1 + df2 * dg2 + df4 * dg4)
    k14 = df4 * (dg1**2 + dg2**2 + dg3**2) - dg4 * (df1 * dg1 + df2 * dg2 + df3 * dg3)
    k21 = dg1 * (df2**2 + df3**2 + df4**2) - df1 * (df2 * dg2 + df3 * dg3 + df4 * dg4)
    k22 = dg2 * (df1**2 + df3**2 + df4**2) - df2 * (df1 * dg1 + df3 * dg3 + df4 * dg4)
    k23 = dg3 * (df1**2 + df2**2 + df4**2) - df3 * (df1 * dg1 + df2 * dg2 + df4 * dg4)
    k24 = dg4 * (df1**2 + df2**2 + df3**2) - df4 * (df1 * dg1 + df2 * dg2 + df3 * dg3)

    #A = [df1,dg1;df2,dg2;df3,dg3;df4,dg4]; M=A'*A; detM=det(M);
    detM = (df1 * dg2 - df2 * dg1)**2 + (df1 * dg3 - df3 * dg1)**2 + (df2 * dg4 - df4 * dg2)**2 + \
        (df3 * dg2 - df2 * dg3)**2 + (df4 * dg1 - df1 * dg4)**2 + (df4 * dg3 - df3 * dg4)**2  # verified!

    # compute fluxes of Strongly Interacting state f** and g**
    fOO = (k11 * b1 + k12 * b2 + k13 * b3 + k14 * b4) / detM
    gOO = (k21 * b1 + k22 * b2 + k23 * b3 + k24 * b4) / detM

    return fOO, gOO

def Euler_IC2d(x, y, input):
    """
    Load the IC of a 1D Riemann classical shock tube problem configuration.
    """
    # Load the IC of a 1D Riemann classical shock tube problem configuration.
    # In the notation we take advantage of the matlab array notation as follows
    #
    #   1.0 +-----------+-----------+
    #       |           |           |
    #       |   reg 2   |   reg 1   |
    #       |           |           |
    #   0.5 +-----------+-----------+
    #       |           |           |
    #       |   reg 3   |   reg 4   |
    #       |           |           |
    #   0.0 +-----------+-----------+
    #      0.0         0.5         1.0
    #
    # prop = [prop_reg1 , prop_reg2 , prop_reg3 , prop_reg4]
    #
    #   r = rho/density
    #   u = velocity in x direction
    #   v = velocity in y direction
    #   p = Pressure
    #
    # Manuel Diaz, NTU, 2014.06.27

    ## Initial Physical Properties per case:
    match input:
        case 1:  # Configuration 1
            print('Configuration 1')
            p = [  1.0 ,  0.4    ,  0.0439 ,  0.15   ]
            r = [  1.0 ,  0.5197 ,  0.1072 ,  0.2579 ]
            u = [  0.0 , -0.7259 , -0.7259 ,  0.0    ]
            v = [  0.0 , -0.0    , -1.4045 , -1.4045 ]
        case 2:  # Configuration 2
            print('Configuration 2')
            p = [  1.0 ,  0.4    ,  1.0    ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  1.0    ,  0.5197 ]
            u = [  0.0 , -0.7259 , -0.7259 ,  0.0    ]
            v = [  0.0 ,  0.0    , -0.7259 , -0.7259 ]
        case 3:  # Configuration 3
            print('Configuration 3')
            p = [  1.5 ,  0.3    ,  0.029 ,  0.3    ]
            r = [  1.5 ,  0.5323 ,  0.138 ,  0.5323 ]
            u = [  0.0 ,  1.206  ,  1.206 ,  0.0    ]
            v = [  0.0 ,  0.0    ,  1.206 ,  1.206  ]
        case 4:  # Configuration 4
            print('Configuration 4')
            p = [  1.1,  0.35   ,  1.1    ,  0.35   ]
            r = [  1.1,  0.5065 ,  1.1    ,  0.5065 ]
            u = [  0.0,  0.8939 ,  0.8939 ,  0.0    ]
            v = [  0.0,  0.0    ,  0.8939 ,  0.8939 ]
        case 5:  # Configuration 5
            print('Configuration 5')
            p = [  1.0 ,  1.0   ,  1.0   ,  1.0   ]
            r = [  1.0 ,  2.0   ,  1.0   ,  3.0   ]
            u = [ -0.75, -0.75  ,  0.75  ,  0.75  ]
            v = [ -0.5 ,  0.5   ,  0.5   , -0.5   ]
        case 6:  # Configuration 6
            print('Configuration 6')
            p = [  1.0  ,  1.0  ,  1.0  ,  1.0  ]
            r = [  1.0  ,  2.0  ,  1.0  ,  3.0  ]
            u = [  0.75 ,  0.75 , -0.75 , -0.75 ]
            v = [ -0.5  ,  0.5  ,  0.5  , -0.5  ]
        case 7:  # Configuration 7
            print('Configuration 7')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  0.8 ,  0.5197 ]
            u = [  0.1 , -0.6259 ,  0.1 ,  0.1    ]
            v = [  0.1 ,  0.1    ,  0.1 , -0.6259 ]
        case 8:  # Configuration 8
            print('Configuration 8')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5197 ,  1.0    ,  0.8 ,  1.0    ]
            u = [  0.1    , -0.6259 ,  0.1 ,  0.1    ]
            v = [  0.1    ,  0.1    ,  0.1 , -0.6259 ]
        case 9:  # Configuration 9
            print('Configuration 9')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.039  ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  0.3 , -0.3 , -0.8133 , -0.4259 ]
        case 10:  # Configuration 10
            print('Configuration 10')
            p = [  1.0    ,  1.0    ,  0.3333 ,  0.3333 ]
            r = [  1.0    ,  0.5    ,  0.2281 ,  0.4562 ]
            u = [  0.0    ,  0.0    ,  0.0    ,  0.0    ]
            v = [  0.4297 ,  0.6076 , -0.6076 , -0.4297 ]
        case 11:  # Configuration 11
            print('Configuration 11')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5313 ,  0.8 ,  0.5313 ]
            u = [  0.1 ,  0.8276 ,  0.1 ,  0.1    ]
            v = [  0.0 ,  0.0    ,  0.0 ,  0.7276 ]
        case 12:  # Configuration 12
            print('Configuration 12')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5313 ,  1.0    ,  0.8 ,  1.0    ]
            u = [  0.0    ,  0.7276 ,  0.0 ,  0.0    ]
            v = [  0.0    ,  0.0    ,  0.0 ,  0.7276 ]
        case 13:
            print('Configuration 13')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5313 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [ -0.3 ,  0.3 ,  0.8145 ,  0.4276 ]
        case 14:  # Configuration 14
            print('Configuration 14')
            p = [  8.0    ,  8.0    ,  2.6667 ,  2.6667 ]
            r = [  2.0    ,  1.0    ,  0.4736 ,  0.9474 ]
            u = [  0.0    ,  0.0    ,  0.0    ,  0.0    ]
            v = [ -0.5606 , -1.2172 ,  1.2172 ,  1.1606 ]
        case 15:  # Configuration 15
            print('Configuration 15')
            p = [  1.0 ,  0.4    ,  0.4 ,  0.4    ]
            r = [  1.0 ,  0.5197 ,  0.8 ,  0.5313 ]
            u = [  0.1 , -0.6259 ,  0.1 ,  0.1    ]
            v = [ -0.3 , -0.3    , -0.3 ,  0.4276 ]
        case 16:  # Configuration 16
            print('Configuration 16')
            p = [  0.4    ,  1.0    ,  1.0 ,  1.0    ]
            r = [  0.5313 ,  1.0222 ,  0.8 ,  1.0    ]
            u = [  0.1    , -0.6179 ,  0.1 ,  0.1    ]
            v = [  0.1    ,  0.1    ,  0.1 ,  0.8276 ]
        case 17:  # Configuration 17
            print('Configuration 17')
            p = [  1.0 ,  1.0 , 0.4    ,  0.4     ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [ -0.4 , -0.3 ,  0.2145 , -1.1259 ]
        case 18:  # Configuration 18
            print('Configuration 18')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  1.0 , -0.3 ,  0.2145 ,  0.2741 ]
        case 19:  # Configuration 19
            print('Configuration 19')
            p = [  1.0 ,  1.0 ,  0.4    ,  0.4    ]
            r = [  1.0 ,  2.0 ,  1.0625 ,  0.5197 ]
            u = [  0.0 ,  0.0 ,  0.0    ,  0.0    ]
            v = [  0.3 , -0.3 ,  0.2145 , -0.4259 ]
        case 'sod_x':  # Sod's shock tube in the x-direction (2-d test)
            print('Sods Shocktube in the x-direction (2-d test)')
            p = [  0.1   ,  1 ,  1 ,  0.1   ]
            r = [  0.125 ,  1 ,  1 ,  0.125 ]
            u = [  0     ,  0 ,  0 ,  0     ]
            v = [  0     ,  0 ,  0 ,  0     ]
        case 'sod_y':  # Sod's shock tube in the y-direction (2-d test)
            print('Sods Shocktube in the y-direction (2-d test)')
            p = [  1 ,  1 ,  0.1   ,  0.1   ]
            r = [  1 ,  1 ,  0.125 ,  0.125 ]
            u = [  0 ,  0 ,  0     ,  0     ]
            v = [  0 ,  0 ,  0     ,  0     ]
        case 'constant':  # Constant state (2-d test)
            print('Constant state (2-d test)')
            p = [  0.1   ,  0.1   ,  0.1   ,  0.1   ]
            r = [  0.125 ,  0.125 ,  0.125 ,  0.125 ]
            u = [  0     ,  0     ,  0     ,  0     ]
            v = [  0     ,  0     ,  0     ,  0     ]
        case _:
            raise Exception(f"specified case '{input}' not available")

    ## Print configuration of selected IC
    print('\n')
    print('          reg 1   reg 2   reg 3   reg 4')
    print(f'density : {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f}')
    print(f'  x-vel : {u[0]:7.4f} {u[1]:7.4f} {u[2]:7.4f} {u[3]:7.4f}')
    print(f'  y-vel : {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} {v[3]:7.4f}')
    print(f'Presure : {p[0]:7.4f} {p[1]:7.4f} {p[2]:7.4f} {p[3]:7.4f}')
    print('\n')

    reg1 = (x >= 0.5) & (y >= 0.5);  # region 1
    reg2 = (x < 0.5) & (y >= 0.5); # region 2
    reg3 = (x < 0.5) & (y < 0.5); # region 3
    reg4 = (x >= 0.5) & (y < 0.5); # region 4

    # Initial Condition for our 2D domain
    r_0 = r[0]*reg1 + r[1]*reg2 + r[2]*reg3 + r[3]*reg4; # Density, rho
    u_0 = u[0]*reg1 + u[1]*reg2 + u[2]*reg3 + u[3]*reg4; # velocity in x
    v_0 = v[0]*reg1 + v[1]*reg2 + v[2]*reg3 + v[3]*reg4; # velocity in y
    p_0 = p[0]*reg1 + p[1]*reg2 + p[2]*reg3 + p[3]*reg4; # temperature.

    return r_0, u_0, v_0, p_0