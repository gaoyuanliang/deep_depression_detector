# Deep Depression Detector

Deep learning model of depression detection from activity sensor data

<img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/dgsadgsgs.gif" width="800">

Same sample activity data waves and their corresponding outputs

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
pip3 install -r requirements.txt
```

download my deep depression detection model file from ```https://drive.google.com/file/d/1mpNJOdKB9JWFfjzqnX0zTwi_-cYh-XS3/view?usp=sharing```

## Usage

download the sample activity level data from https://datasets.simula.no/depresjon/

```bash
wget https://datasets.simula.no/depresjon/data/depresjon-dataset.zip
unzip depresjon-dataset.zip
```

after unzip if you will see folders and activity data csv file

<img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/WX20200901-181410%402x.png" width="300">  <img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/WX20200901-181426%402x.png" width="300">


in this data set, each person's activity level is stored in a csv file. Given the csv file path, the program will read the data and do the preditction. The output is a prediction with a confidence score

```python
>>> from jessica_deep_depression_detector import deep_depression_detector
>>> 
>>> deep_depression_detector('data/control/control_2.csv')
{'prediction': 'nondepressed', 'confidence': 0.97963333}
>>> 
>>> deep_depression_detector('data/condition/condition_11.csv')
{'prediction': 'depressed', 'confidence': 0.9602384}
```
## Model Structure

<img src="https://raw.githubusercontent.com/gaoyuanliang/deep_depression_detector/master/model.png" width="600">
