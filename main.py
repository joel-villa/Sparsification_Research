import matplotlib.pyplot as plt
from SSGetter import SSGetter
from Tester import Tester

def plot(X, Y, labels, ns):
    """
    Generate plots

    X - A 2D array of x values
    Y - A 2D array of y values
    """
    title = "Sparsification Behavior"
    x_label = "s"
    y_label = rf"$||e - \tilde e||$"
    for x, y, lbl, n in zip(X, Y, labels, ns):
        t = title + " of " + lbl
        plt.plot(x, y)
        plt.title(t + f" (n = {n})")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend()
        file_name = t.replace(" ", "_")
        plt.savefig("plots/" + lbl  + ".svg")
        plt.show()

if __name__ == '__main__':
    tester = Tester()

    # big_ssgetter = SSGetter(True, row_bounds=(17755,100000))
    small_ssgetter = SSGetter(True, row_bounds=(100,10000))

    # big_mats = big_ssgetter.get_next(5)
    small_mats = small_ssgetter.get_next(5)

    S = []
    D = []
    names = []
    ns = []

    for name, A in small_mats.items():
        print(name)
        n, _ = A.shape
        ss, _, diff = tester.test_s_behavior(A)
        D.append(diff)
        S.append(ss)
        ns.append(n)
        names.append(name)

    plot(S, D, names, ns)
