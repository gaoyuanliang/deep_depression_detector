# Deep Depression Detector

Deep learning model of depression detection from activity sensor data

<table>
  <thead>
    <tr>
      <th>Input activity level data</th>
      <th>Output</th>
    </tr>
  </thead>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/condition_11.png" width="600">
    </td>
    <td>
      <pre>
{
  'prediction': 'depressed', 
  'confidence': 0.9602384
}
</pre>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/control_2.png" width="600">
    </td>
    <td>
      <pre>
{
  'prediction': 'nondepressed', 
  'confidence': 0.97963333
}
</pre>
    </td>
  </tr>
</table>

## Instillation

```bash
git clone https://github.com/gaoyuanliang/deep_depression_detector.git
cd deep_depression_detector
pip3 install -r 
```

## Usage

```python
>>> from jessica_deep_depression_dertector import deep_depression_detector
>>> 
>>> deep_depression_detector('data/control/control_2.csv')
{'prediction': 'nondepressed', 'confidence': 0.97963333}
>>> 
>>> deep_depression_detector('data/condition/condition_11.csv')
{'prediction': 'depressed', 'confidence': 0.9602384}
```
## Model Structure

<img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/model.png" width="600">
