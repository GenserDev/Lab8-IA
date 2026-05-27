"""
Ejercicio 1 – Distribución conjunta P(X1, X2, X3)
"""

joint = {
    (0, 0, 0): 0.05,
    (1, 0, 0): 0.10,
    (0, 1, 0): 0.40,
    (1, 1, 0): 0.10,
    (0, 0, 1): 0.10,
    (1, 0, 1): 0.05,
    (0, 1, 1): 0.20,
    (1, 1, 1): 0.00,
}


def marginal(**conditions):
    """Suma sobre todos los valores que satisfacen las condiciones dadas."""
    return sum(
        p for (x1, x2, x3), p in joint.items()
        if all(
            (x1 == v if k == "x1" else x2 == v if k == "x2" else x3 == v)
            for k, v in conditions.items()
        )
    )


def conditional(numer_conds, denom_conds):
    return marginal(**numer_conds) / marginal(**denom_conds)


# a) P(X1=1, X2=0)
a = marginal(x1=1, x2=0)

# b) P(X3=0)
b = marginal(x3=0)

# c) P(X2=1 | X3=1)
c = conditional({"x2": 1, "x3": 1}, {"x3": 1})

# d) P(X1=0 | X2=1, X3=1)
d = conditional({"x1": 0, "x2": 1, "x3": 1}, {"x2": 1, "x3": 1})

# e) P(X1=0, X2=1 | X3=1)
e = conditional({"x1": 0, "x2": 1, "x3": 1}, {"x3": 1})


if __name__ == "__main__":
    print("=" * 45)
    print("Ejercicio 1 – Distribución conjunta")
    print("=" * 45)
    print(f"a) P(X1=1, X2=0)             = {a:.4f}")
    print(f"b) P(X3=0)                   = {b:.4f}")
    print(f"c) P(X2=1 | X3=1)            = {c:.4f}  (= 4/7)")
    print(f"d) P(X1=0 | X2=1, X3=1)     = {d:.4f}")
    print(f"e) P(X1=0, X2=1 | X3=1)     = {e:.4f}  (= 4/7)")
