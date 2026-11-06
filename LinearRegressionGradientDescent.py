import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def f(w1, w2, x, y):
    return (x - w1) ** 2 + y * (w2 - w1 ** 2) ** 2

def grad_f(w1, w2, x, y):
    r = w2 - w1 ** 2
    d_w1 = -2.0 * (x - w1) - 4.0 * y * w1 * r
    d_w2 = 2.0 * y * r
    return d_w1, d_w2

def gradient_descent(x, y, w_init, alpha, max_iters=200000, eps=1e-6, record_every=0):
    w1, w2 = float(w_init[0]), float(w_init[1])
    path = [(w1, w2)] if record_every else None
    status = "stopped at max iterations"
    it = 0

    for it in range(1, max_iters + 1):
        g1, g2 = grad_f(w1, w2, x, y)
        w1 -= alpha * g1
        w2 -= alpha * g2

        if not (math.isfinite(w1) and math.isfinite(w2)) or abs(w1) > 1e8 or abs(w2) > 1e8:
            status = "diverged"
            break

        if record_every and it % record_every == 0:
            path.append((w1, w2))

        if math.hypot(g1, g2) <= eps:
            status = "converged"
            break

    if record_every and status != "diverged":
        path.append((w1, w2))

    final_f = f(w1, w2, x, y) if status != "diverged" else float("nan")
    return {
        "w": (w1, w2),
        "f": final_f,
        "iters": it,
        "status": status, "path": path, }

def part_b():
    print("Part (b)")
    x, y = 1, 100
    w_init = (-1.5, 2.0)
    alpha = 0.001
    res = gradient_descent(x, y, w_init, alpha)
    print("x =", x, ", y =", y, ", start =", w_init, ", learning rate =", alpha)
    print("w1 =", res["w"][0], ", w2 =", res["w"][1], ", f =", res["f"],
          ", iterations =", res["iters"], ",", res["status"])
    print()


EXPERIMENTS = {
    (1, 100): {
        "starts": [(0.0, 0.0), (-1.5, 2.0), (2.0, 2.0), (-1.0, -1.0)],
        "alphas": [0.0001, 0.001, 0.002, 0.005],
        "max_iters": 200000,
    },
    (5, 50): {
        "starts": [(0.0, 0.0), (3.0, 10.0), (6.0, 30.0), (4.0, 20.0)],
        "alphas": [0.00005, 0.0001, 0.00019, 0.00025],
        "max_iters": 3000000,
    },
}


def part_c():
    print("Part (c)")
    for (x, y), cfg in EXPERIMENTS.items():
        print()
        print("x =", x, ", y =", y)
        for alpha in cfg["alphas"]:
            print("Learning rate =", alpha)
            for start in cfg["starts"]:
                res = gradient_descent(x, y, start, alpha,
                                       max_iters=cfg["max_iters"])
                print("  start =", start, ": w1 =", round(res["w"][0], 5),
                      ", w2 =", round(res["w"][1], 5),
                      ", f =", f"{res['f']:.2e}",
                      ", iterations =", res["iters"], ",", res["status"])
    print()

def make_plots():
    matplotlib.use("Agg")

    for (x, y), cfg in EXPERIMENTS.items():
        # plot 1: paths of gradient descent on top of a contour plot of f
        if x == 1:
            alpha = 0.001
            W1, W2 = np.meshgrid(np.linspace(-2, 2.5, 200), np.linspace(-1.5, 4, 200))
        else:
            alpha = 0.00019
            W1, W2 = np.meshgrid(np.linspace(-1, 7, 200), np.linspace(-2, 35, 200))
        Z = f(W1, W2, x, y)

        plt.figure()
        plt.contour(W1, W2, Z, levels=np.logspace(-1, 4, 20))
        for start in cfg["starts"]:
            res = gradient_descent(x, y, start, alpha, max_iters=cfg["max_iters"],
                                   record_every=100)
            path = np.array(res["path"])
            plt.plot(path[:, 0], path[:, 1], label="start " + str(start))
        plt.plot(x, x ** 2, "kx", markersize=10, label="minimum")
        plt.xlabel("w1")
        plt.ylabel("w2")
        plt.title("GD paths, x=" + str(x) + ", y=" + str(y) + ", alpha=" + str(alpha))
        plt.legend()
        plt.savefig("paths_x" + str(x) + "_y" + str(y) + ".png")
        plt.close()

        # plot 2: f vs iterations for each learning rate, starting from (0, 0)
        plt.figure()
        for alpha in cfg["alphas"]:
            w1, w2 = 0.0, 0.0
            iters, values = [], []
            for t in range(0, cfg["max_iters"], 100):
                iters.append(t)
                values.append(f(w1, w2, x, y))
                for _ in range(100):
                    g1, g2 = grad_f(w1, w2, x, y)
                    w1 = w1 - alpha * g1
                    w2 = w2 - alpha * g2
                if abs(w1) > 1e8 or abs(w2) > 1e8:
                    break
            plt.plot(iters, values, label="alpha = " + str(alpha))
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("f(w1, w2)")
        plt.title("f vs iterations, x=" + str(x) + ", y=" + str(y) + ", start (0,0)")
        plt.legend()
        plt.savefig("lr_curves_x" + str(x) + "_y" + str(y) + ".png")
        plt.close()

def main():
    part_b()
    part_c()
    make_plots()

if __name__ == "__main__":
    main()