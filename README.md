# deep_depression_detector
Deep learning model of depression detection from activity sensor data

## Usage of the deep depression detector

```python
>>> from jessica_deep_depression_dertector import deep_depression_detector
>>> 
>>> deep_depression_detector('data/control/control_2.csv')
{'prediction': 'nondepressed', 'confidence': 0.97963333}
>>> 
>>> deep_depression_detector('data/condition/condition_11.csv')
{'prediction': 'depressed', 'confidence': 0.9602384}
```
