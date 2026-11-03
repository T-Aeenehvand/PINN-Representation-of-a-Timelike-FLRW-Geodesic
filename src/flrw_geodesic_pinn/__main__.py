from .physics import conformal_targets

if __name__ == "__main__":
    print("FLRW geodesic PINN package installed. See README.md and scripts/ for experiments.")
    print("At eta=0.5, dx/deta and dlambda/deta =", conformal_targets(0.5))
