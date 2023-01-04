# Introduction
This program implements the Weighted Sampling Algorithm for radio map reconstruction in the Final Year Project of Zixian Jin.

# How to Run the Program
Before running the program, make sure the following packages have been installed to your Python:
```bash
tensorflow
scipy
cvxpy
cvxopt
matplotlib
pandas
joblib
sklearn
opencv-python
```

Then run the "main.py" without inputing any arguments. You will see the plotted figures, which are also presented in the final report.


# Reference
The code for acquiring radio map datasets and implementing Kriging interpolation is based on the work of https://github.com/fachu000/deep-autoencoders-cartography.git. Below is the reference of related article:

@article{teganya2020deepcompletion,
  title={Deep Completion Autoencoders for Radio Map Estimation},
  author={Teganya, Yves and Romero, Daniel},
  journal={arXiv preprint arXiv:2005.05964},
  year={2020}
}

