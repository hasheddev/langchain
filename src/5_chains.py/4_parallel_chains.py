# tasks run side by side
# Branches based on rules
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from models import qwen_model, gemini_model

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="Generate short and simple note from the following topic: {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text: {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and question answers into a single document ```{notes}```, ```{qa}```",
    input_variables=["notes", "qa"],
)

runnable_chain = RunnableParallel(
    {"notes": prompt1 | qwen_model | parser, "qa": prompt2 | gemini_model | parser}
)

merge_chain = prompt3 | qwen_model | parser

svm_notes = """
Support Vector Machines (SVM): Comprehensive Study Notes

Support Vector Machines (SVMs) are powerful, versatile supervised machine learning models primarily used for classification tasks, though they can also be applied to regression (Support Vector Regression - SVR).

1. Core Definitions and Terminology
- Hyperplane: A decision boundary that separates data points of different classes in an N-dimensional space. In 2D space, a hyperplane is a line; in 3D, it is a plane; in higher dimensions, it is a hyperplane. Its mathematical equation is w · x + b = 0, where w is the weight vector, x is the input feature vector, and b is the bias.
- Support Vectors: The data points lying closest to the decision boundary (hyperplane). They alone define the position and orientation of the hyperplane. Removing any other data points leaves the hyperplane unchanged, whereas removing a support vector shifts the boundary.
- Margin: The perpendicular distance between the optimal hyperplane and the closest data points (support vectors) from either class. SVMs aim to maximize this margin to ensure high generalization performance on unseen data.

2. How SVM Works: Linear Separation
When data is linearly separable, the goal of SVM is to find the Maximum-Margin Hyperplane (MMH).
- Classification Rule: A new data point x is classified based on which side of the hyperplane it falls on, determined by the sign of w · x + b.
- Hard Margin vs. Soft Margin:
  * Hard Margin SVM: Assumes the data is strictly linearly separable with zero tolerance for misclassifications. It is prone to overfitting if outliers exist.
  * Soft Margin SVM: Introduces a regularization parameter (C) and slack variables (xi) to allow a small number of misclassifications or points within the margin, yielding a more robust model in noisy datasets.

3. Handling Non-Linear Data: The Kernel Trick
Real-world datasets are rarely linearly separable in their original input space. SVMs use Kernels—mathematical functions that compute the dot product of data in a higher-dimensional space without transforming the data explicitly. This is known as the Kernel Trick.
Common Kernels:
- Linear Kernel: Best for text classification or high-dimensional datasets where data is already linearly separable.
- Polynomial Kernel: Accounts for feature interactions up to a specified degree.
- Radial Basis Function (RBF) Kernel: Maps data into an infinite-dimensional space; highly effective for complex, non-linear boundaries.
- Sigmoid Kernel: Often used in neural network-like architectures.

4. Advantages and Limitations of SVMs
Advantages:
- Effective in High-Dimensional Spaces: Performs exceptionally well when the number of features exceeds the number of samples (e.g., text classification, genomics).
- Memory Efficient: Uses only a subset of training data (support vectors) in the decision function, conserving memory during inference.
- Versatile: Custom kernel functions allow adaptation to complex decision boundaries.
Limitations:
- Poor Scalability on Large Datasets: Training time scales cubically or quadratically with dataset size, making it computationally expensive for massive modern datasets.
- Sensitive to Noise and Outliers: Severe outliers can heavily distort the maximum-margin hyperplane in hard or poorly-regularized soft margin setups.
- Lack of Direct Probability Estimates: SVM outputs distances to the hyperplane rather than direct class probabilities (though probabilities can be approximated using Platt scaling).
"""

chain = runnable_chain | merge_chain
result = chain.invoke({"text": svm_notes})

print(result)
