# Credit Scoring Data Modeling

## Overview
The financial firm "Prêt à dépenser" is a consumer credit company for people with few or no credit history.

In order to offer more transparency regarding its credit granting decisions, the company wants to develop an interactive dashboard based on a machine learning model scoring the default probability of a given client. This model should be based on a variety of data (behavioral, from other financial institutions...).

This repository contains several ordered notebooks presenting the steps taken to achieve the modeling of the input data : 
- Exploratory Data Analysis
- Data Assembly
- Balancing Method & Algorithm Selection
- Feature selection
- Hyperparameter Tuning & Final Model Explainability
- Data Assembly for the Dashboard

The resulting model outputs the credit scoring for a given client on a scale from 0 to 100, 0 being the best value (0 risk of default), 100 being the worst value (no chance the client will pay back its credit).


## Requirements
See requirements.txt


## Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk), and place the files under Notebooks/Resources/datasets/origin/.
2. Run the following in your terminal to install all required libraries : 
```bash
pip3 install -r requirements.txt
```
3. Run each notebook one after the other, following the order indicated by the digits in each notebook's name.

For a complete overview of the modeling approach, please see the methodology note.

## Credit
A big thank you to Will Koehrsen, whose [notebooks](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction) were a huge help and inspiration for tackling this problem.
