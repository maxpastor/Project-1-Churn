# Project 1: Churn
 Project 1 of the **1 Model a Week** series. This one tackles customer churn. 

 ## Disclamer:
Even though I work at AWS, this code is a personal project and is not developped or provided by/on-behalf of AWS. 
The code might have errors or might not follow best practices. 

Running this code on your AWS Account will trigger costs (Compute, Storage and Networking) so use at your own risk. 

## How-To use:

The main notebook contains a simple Feed Forward model that is supposed to output if a customer is going to churn or not. 
The goal is not to have extremely good results, but to show how to implement such a solution. If you fancy experimenting with XGboost to solve this problem, you will see that it actually performs better out of the box... And that's alright. 

The notebook is in 3 Parts (I will document them more later, or at least I'll try):

- 1: We run the training on the Notebook (not best practice but good for quick experimentation and development)
- 2: We run the training using a Sagemaker Training Job on a simple CPU machine (no need for a fancy GPU)
- 3: We run a Hyperparameter tuning Job on Sagemaker where we modulate the learning rate, num of neurons per layer, dropout and number of Epochs, to see if we get a better AUC. 

The code does not have at this moment any Inference code. It would not be so difficult to develop but it was not my primary focus. 
It doesn't even save the model. 
So you will see how the training did but you wont be able to do anything with it at the moment. 

I will implement that last bit when I have the time. 

Feel free to suggest edits. The main goal of this repo and this new series is to force me to write more Pytorch code, as I was a Tensorflow instructor in the past, I need to get more familiar with it. 

XGboost instructions and data can be found here: https://sagemaker-examples.readthedocs.io/en/latest/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.html

Cheers !


