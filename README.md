# Math Solving
A language model capable of answering math questions in alignment with the Vietnamese Education Program.  
This model currently attains an accuracy score of **0.689**, reflecting a moderate performance level. However, the model still has a lot of potential for improvement.
## Quick Start
Clone this project and install the required packages:
```
git clone https://github.com/QuagHien/math-solving.git
pip install -r math-solving/requirements.txt
```
## Data preprocessing
Organize dataset folder as follows:
```
python3 math-solving/preprocessing.py
 ```
## Dataloader
Translation data to English:
```
python3 math-solving/dataloader.py
```
## Train and predict
Model: microsoft/deberta-v3-large
```
python3 math-solving/train.py
```

