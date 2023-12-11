# Math Solving
A language model capable of answering math questions in alignment with the Vietnamese Education Program.
## Quick Start
Clone this project and install the required packages:
```
git clone https://github.com/QuagHien/advertisingbanner-generation.git
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
## Model: microsoft/deberta-v3-large
## Train and predict
```
python3 math-solving/train.py
```
## Model evaluation
This model currently attains an accuracy score of 0.635, reflecting a moderate performance level. However, there exists considerable untapped potential for enhancement and refinement in various aspects.
