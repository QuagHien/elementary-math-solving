# Elementary Math Solving with Text Classification
The language model is capable of answering Vietnamese elementary math questions with Text Classification.  
## Quick Start
Clone this project and install the required packages:
```
git clone https://github.com/QuagHien/math-solving.git
pip install -r math-solving/requirements.txt
```
## Data
Organize data folder as example data:
```
gdown --id 1oTtVqsHoUQL9Q_1OdlidwcTr4tX59YvX
gdown --id 1Pc3TMNJK5Vs_gvoF2aWcIhYG3l0bcEUB
```
## Data preprocessing
Organize data folder as follows:
```
python3 math-solving/preprocessing.py
 ```
## Dataloader
Translation data to English:
```
python3 math-solving/dataloader.py
```
## Train and predict
**Model: OpenAssistant/reward-model-deberta-v3-large-v2**
```
python3 math-solving/train.py
```

