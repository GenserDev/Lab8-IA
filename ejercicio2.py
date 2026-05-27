"""
Ejercicio 2 – Red bayesiana en cadena: R -> T -> L
  R: Raining  |  T: Traffic  |  L: Late for class
"""

# P(R)
P_R = {1: 0.1, 0: 0.9}

# P(T | R)
P_T_given_R = {
    (1, 1): 0.8,  # P(T=1 | R=1)
    (0, 1): 0.2,  # P(T=0 | R=1)
    (1, 0): 0.1,  # P(T=1 | R=0)
    (0, 0): 0.9,  # P(T=0 | R=0)
}

# P(L | T)
P_L_given_T = {
    (1, 1): 0.3,  # P(L=1 | T=1)
    (0, 1): 0.7,  # P(L=0 | T=1)
    (1, 0): 0.1,  # P(L=1 | T=0)
    (0, 0): 0.9,  # P(L=0 | T=0)
}


def P_T(t):
    return sum(P_T_given_R[(t, r)] * P_R[r] for r in (0, 1))


def P_L(l):
    return sum(P_L_given_T[(l, t)] * P_T(t) for t in (0, 1))


# a) P(T=1 | R=0)  — lectura directa de la CPT
a = P_T_given_R[(1, 0)]

# b) P(T=1)  — suma sobre R
b = P_T(1)

# c) P(L=1 | R=0, T=1)  — T d-separa L de R en la cadena
c = P_L_given_T[(1, 1)]

# d) P(L=1 | T=1)
d = P_L_given_T[(1, 1)]

# e) P(L=1 | R=0)  — suma sobre T
e = sum(P_L_given_T[(1, t)] * P_T_given_R[(t, 0)] for t in (0, 1))

# f) P(T=1 | L=0)  — Bayes: P(L=0|T=1)P(T=1) / P(L=0)
p_l0 = P_L(0)
f = P_L_given_T[(0, 1)] * P_T(1) / p_l0


if __name__ == "__main__":
    print("=" * 45)
    print("Ejercicio 2 – Red bayesiana R -> T -> L")
    print("=" * 45)
    print(f"a) P(T=1 | R=0)          = {a:.4f}")
    print(f"b) P(T=1)                = {b:.4f}")
    print(f"c) P(L=1 | R=0, T=1)    = {c:.4f}  (igual a d, T bloquea R)")
    print(f"d) P(L=1 | T=1)         = {d:.4f}")
    print(f"e) P(L=1 | R=0)         = {e:.4f}")
    print(f"f) P(T=1 | L=0)         = {f:.4f}")
