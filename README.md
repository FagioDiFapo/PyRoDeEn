# Approximate Riemann Solvers – Python Port and Extensions

This repository is a Python-based port and extension of a collection of finite difference (FD) and finite volume (FV) solvers originally developed in MATLAB. The solvers implement a variety of numerical schemes — including MUSCL, WENO, and THINC-BVD — to solve the Euler equations using approximate Riemann solvers.

The original MATLAB scripts were developed by a former Cranfield University student (author unknown), who has made them publicly available. This Python version is being developed as part of a Master's level Internal Research Project in Astronautics and Space Engineering at Cranfield University, with a focus on applying these solvers to simulate Rotating Detonation Engines (RDEs).

## Project Goals

- **Port** key 1D and 2D MATLAB solvers to Python.
- **Modularize** the solver structure to enable reuse and extension.
- **Implement** chemical kinetics (e.g., 10-step C₂H₄/O₂ mechanism).
- **Simulate** a 2D Rotating Detonation Engine using reduced-order modeling.

## Attribution

The original MATLAB source code was created and shared by:

**Manuel A. Diaz**  
GitHub: GitHub: [@wme7](https://github.com/wme7)  
Original repo (archived/forked): [Approximate Riemann Solvers](https://github.com/wme7/ApproximateRiemannSolvers)

> *“This repo is my personal collection of finite difference (FD) and finite volume (FV) Riemann solvers using MUSCL and WENO schemes. These solvers are written as short Matlab scripts and they are now publicly available as I've moved to another field of CFD.”* — M.A. Diaz

All original references are preserved in the Python code and/or this repository.

## Licensing and Attribution

This work builds upon MATLAB code developed at Cranfield University and publicly shared by Manuel Diaz. Much of the work here is also based on the insights and further development by GitHub: [https://github.com/wme7](https://github.com/rhann-09) If you are the original author and would like to be credited explicitly, please open an issue or contact me.

All Python code written for this project is © Genis Bonet Garcia, 2025, under MIT.

---

> **Disclaimer:** This repository is a research tool and is not guaranteed to be production-ready. Use at your own discretion.
