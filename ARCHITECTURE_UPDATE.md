# Machine Learning Architecture Update & Explanation

This document explains the updates made to our vulnerability detection pipeline, specifically detailing why we changed the original models and how our new hybrid model (Stacking) works. This is useful for team presentations and understanding the core machine learning concepts behind the project.

## 1. What We Removed and Why

**Removed:** VGG16, AlexNet, and LSTM models (1D Convolutional and Recurrent Neural Networks).

**Reason for Removal (The Tabular Data Problem):**
The original code attempted to use VGG16, AlexNet, and LSTM on our dataset. However, these specific Deep Learning models are designed for **sequential or spatial data** where the *order* of the data matters.
- For example, an image is spatial: if you shuffle the pixels randomly, the image is destroyed. A CNN (like VGG16) looks at pixels next to each other to find shapes.
- A sentence is sequential: "Dog bites man" is different from "Man bites dog". An LSTM reads words in order.

Our dataset is a CSV table of 44 complexity metrics (Lines of Code, Cyclomatic Complexity, etc.). This is called **Tabular Data**. In our table, the order of the columns is completely random (e.g., the `CC` column is next to `CCL` just by chance). If we use a CNN on this table, it tries to find "patterns" across adjacent columns, which is mathematically meaningless. It's like asking an image-recognition AI to read a spreadsheet.

*(Note: In the original research paper, the authors likely used CNNs and LSTMs on the actual JavaScript source code text, not on the CSV of complexity metrics.)*

## 2. What We Replaced It With

We replaced those models with a **Hybrid Stacking Ensemble** that combines traditional Machine Learning (Trees) with correct Deep Learning (MLP).

### The Machine Learning Models (Tree Ensembles)
For Tabular Data, tree-based models are universally considered the state-of-the-art. We are using:
1. **XGBoost (eXtreme Gradient Boosting):** This model builds hundreds of small "Decision Trees" sequentially. Each new tree specifically looks at the mistakes made by the previous trees and tries to correct them. It is extremely fast and highly accurate for CSV data.
2. **Random Forest:** This model builds hundreds of "Decision Trees" in parallel. Each tree looks at a random subset of the data and a random subset of the features. It then averages the decisions of all the trees. This prevents the model from memorizing (overfitting) the training data.

### The Deep Learning Model (MLP)
Instead of a CNN, we introduced a **Multi-Layer Perceptron (MLP)**. An MLP uses standard "Dense" layers. Unlike a CNN, a Dense layer connects every input to every neuron. It looks at all 44 features simultaneously and does not care what order the columns are in. This is the scientifically correct way to apply Deep Learning to tabular data.

## 3. How "Stacking" Works (Combining ML and Deep Learning)

Stacking is an advanced technique where we train multiple different models (called "base learners") and use a final model (called a "meta-learner") to make the ultimate decision.

Here is how our pipeline works:
1. **Base Training:** We train our XGBoost, Random Forest, and MLP models independently on the dataset. 
2. **Voting:** We ask all three models to predict whether a function is vulnerable. They output probabilities (e.g., XGBoost says 80%, Random Forest says 70%, MLP says 90%).
3. **The Meta-Learner:** We feed these three probabilities into a final model (a **Logistic Regression** model). The Logistic Regression model learns *which model to trust the most*. For example, it might learn that if XGBoost and the MLP agree, it should ignore Random Forest. 
4. **Final Decision:** The meta-learner outputs the final prediction.

**Why do this?** 
Machine Learning models (Trees) and Deep Learning models (MLP) learn completely differently. Trees are great at finding hard thresholds (e.g., "If complexity > 10, it's vulnerable"), while Neural Networks are great at finding smooth, complex mathematical relationships. By stacking them, we get the best of both worlds!
