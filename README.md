# 🐚 PyTorch Multiple Regression: Abalone Age Prediction

This project implements a Multiple Regression model using PyTorch to predict the age (number of Rings) of Abalone sea snails based on various physical measurements. The workflow follows the standard PyTorch deep learning pipeline.

![Linear Regression PyTorch](/Abalone.png)

## ⚙️ Prerequisites
To run this notebook, you need the following Python packages. You can install them using pip:

```Bash
!pip install ucimlrepo feature_engine torchmetrics pandas torch numpy matplotlib seaborn scikit-learn --quiet
```

## The key libraries used are:
* `torch`
* `torch.nn`: For building and training the deep learning model.
* `ucimlrepo`: For easily fetching the Abalone dataset.
* `feature_engine`: For one-hot encoding categorical variables.
* `sklearn.metrics`: For calculating the Root Mean Squared Error (RMSE).

## 💾 Dataset and Preprocessing
The model uses the Abalone Dataset from the UCI Machine Learning Repository.

## Data Preparation Steps
1. Data Loading: The dataset is fetched using ucimlrepo.
2. One-Hot Encoding: The categorical feature Sex is one-hot encoded using feature_engine.encoding.OneHotEncoder.
3. Feature Engineering: Second Order Variables are created to potentially capture non-linear relationships:
* Diameter_2 (Diameter squared)
* Height_2 (Height squared)
* Shell_2 (Shell_weight squared)
4. Feature Selection: The features Whole_weight and Length are dropped from the input features to mitigate multicollinearity.
5. Data Transformation: All features are converted to PyTorch tensors and wrapped in a DataLoader for efficient batch processing during training.

## 🧠 Model Architecture and Training

### Model Definition

The model is a simple Fully Connected Neural Network (FCNN) implemented using torch.nn.Sequential.

The architecture consists of:

* An input layer with the number of features in the preprocessed dataset.
* Three hidden layers with ReLU activation functions and widths of 64, 32, and 16 nodes, respectively.
* A final output layer with a single node (for regression) and no activation function.

```Python
class RegressionModel(nn.Module):
    def __init__(self, input_features):
        super(RegressionModel, self).__init__()
        self.layer_1 = nn.Linear(input_features, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        x = self.output_layer(x)
        return x
```

### Training Parameters

* Loss Function: Mean Squared Error (MSE) (nn.MSELoss()), a standard loss for regression problems.
* Optimizer: Adam (optim.Adam) with a learning rate of 0.001.
* Epochs: The model is trained for 100 epochs.

## 📊 Results
The model's performance is evaluated using the Root Mean Squared Error (RMSE). The notebook demonstrates the calculation and application of an exponential transformation to the final predictions for a potential improvement.

```
Metric	Value
Final RMSE	2.105
Improvement	~4.1% (Calculated as (2.196 - 2.105) / 2.196)
```

## ▶️ How to Run

1. Clone this repository.
2. Ensure all prerequisites are installed.
3. Open the Pytorch_Multiple_Regression.ipynb file in a Jupyter Notebook environment (like JupyterLab or Google Colab).
4. Run all cells sequentially to load the data, train the model, and view the final results.
