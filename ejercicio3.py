"""
Ejercicio 3 – Red bayesiana: G -> M -> {B, C}
  G: Green Party elected  |  M: Marijuana legalized
  B: Balanced budget      |  C: Class attendance increases
"""
from fractions import Fraction

# P(G)
P_G = {1: Fraction(1, 10), 0: Fraction(9, 10)}

# P(M | G)
P_M_given_G = {
    (1, 1): Fraction(2, 3),    # P(M=1 | G=1)
    (0, 1): Fraction(1, 3),    # P(M=0 | G=1)
    (1, 0): Fraction(1, 4),    # P(M=1 | G=0)
    (0, 0): Fraction(3, 4),    # P(M=0 | G=0)
}

# P(B | M)
P_B_given_M = {
    (1, 1): Fraction(2, 5),    # P(B=1 | M=1)
    (0, 1): Fraction(3, 5),    # P(B=0 | M=1)
    (1, 0): Fraction(1, 5),    # P(B=1 | M=0)
    (0, 0): Fraction(4, 5),    # P(B=0 | M=0)
}

# P(C | M)
P_C_given_M = {
    (1, 1): Fraction(1, 4),    # P(C=1 | M=1)
    (0, 1): Fraction(3, 4),    # P(C=0 | M=1)
    (1, 0): Fraction(1, 2),    # P(C=1 | M=0)
    (0, 0): Fraction(1, 2),    # P(C=0 | M=0)
}

SIGNS = {1: "+", 0: "-"}


def joint(g, m, b, c):
    return P_G[g] * P_M_given_G[(m, g)] * P_B_given_M[(b, m)] * P_C_given_M[(c, m)]


def fmt(f):
    return f"{SIGNS[f[0]]},{SIGNS[f[1]]},{SIGNS[f[2]]},{SIGNS[f[3]]}"


if __name__ == "__main__":
    print("=" * 55)
    print("Ejercicio 3 – Tabla de probabilidad conjunta P(G,M,B,C)")
    print("=" * 55)
    print(f"{'G':>2} {'M':>2} {'B':>2} {'C':>2}   {'P(G,M,B,C)':>14}")
    print("-" * 55)
    total = Fraction(0)
    for g in (1, 0):
        for m in (1, 0):
            for b in (1, 0):
                for c in (1, 0):
                    p = joint(g, m, b, c)
                    total += p
                    print(
                        f"  {SIGNS[g]:>1}  {SIGNS[m]:>1}  {SIGNS[b]:>1}  {SIGNS[c]:>1}"
                        f"   {str(p):>14}  ({float(p):.6f})"
                    )
    print("-" * 55)
    print(f"{'Total':>36}  {float(total):.6f}")
    print()
    print("Valores faltantes calculados:")
    missing = [
        (1, 1, 1, 0, "(+,+,+,-)"),
        (1, 1, 0, 0, "(+,+,-,-)"),
        (1, 0, 0, 1, "(+,-,-,+)"),
        (0, 1, 1, 1, "(-,+,+,+)"),
        (0, 1, 0, 1, "(-,+,-,+)"),
        (0, 0, 0, 1, "(-,-,-,+)"),
    ]
    for g, m, b, c, label in missing:
        p = joint(g, m, b, c)
        print(f"  P{label} = {p} ~ {float(p):.6f}")
