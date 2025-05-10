# Bank Marketing Classifier

This project implements a machine learning classifier to predict whether a client will subscribe to a term deposit based on bank marketing data.

## Project Overview

The Bank Marketing Classifier uses a Decision Tree algorithm to analyze customer data and predict the likelihood of a customer subscribing to a term deposit. The model achieves an accuracy of approximately 91.81% on the test dataset.

## Dataset

The project uses the Bank Marketing Dataset, which contains the following files:
- `bank-additional-full.csv` - Main dataset used for training
- `bank-additional.csv` - Additional dataset
- `bank.csv` - Original dataset
- `bank-full.csv` - Full version of the original dataset

The dataset includes various features such as:
- Customer demographics (age, job, marital status, education)
- Banking information (default, housing, loan)
- Contact information (contact type, month, day of week)
- Campaign data (duration, campaign, pdays, previous, poutcome)
- Economic indicators (employment variation rate, consumer price index, etc.)

## Features

- Data preprocessing with label encoding and standardization
- Decision Tree classification model
- Model evaluation with accuracy metrics and confusion matrix
- Visualization of results using matplotlib and seaborn
- Interactive decision tree visualization

## Requirements

To run this project, you need the following Python packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Karthikprasadm/PRODIGY_DS_03.git
```

2. Navigate to the project directory:
```bash
cd PRODIGY_DS_03
```

3. Run the classifier:
```bash
python bank_marketing_classifier.py
```

## Model Performance

The model shows the following performance metrics:
- Overall Accuracy: 91.81%
- For non-subscribers (class 0):
  - Precision: 94%
  - Recall: 97%
- For subscribers (class 1):
  - Precision: 68%
  - Recall: 52%

## Visualizations

The script generates two main visualizations:
1. Confusion Matrix - Shows the distribution of correct and incorrect predictions
2. Decision Tree - Visualizes the decision-making process of the model

## Author

Karthik Prasad M

## License

This project is open source and available under the MIT License. 