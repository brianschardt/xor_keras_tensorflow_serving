# XOR Keras Trained Model
### Exported Model configured for Tensorflow Serving

There is no real example of a simple model that was trained using 
keras and then exported to in a format specifically for 
Tensorflow Serving. So for the sake of clarity I wanted to use the simplest 
example I could think of, which is the XOR logic gate. https://en.wikipedia.org/wiki/XOR_gate




### Getting Environment Set up
#### Pipenv
I am using pipenv in order to standardize environments, kind or like the famous NPM for node

https://docs.pipenv.org/
```angular2html
pip install pipenv
```
or if you are using mac install with homebrew
```angular2html
brew install pipenv
```

#### Don't want to use Pipenv
If you do not want to use **pipenv** then you must install these dependencies
You must have tensorflow keras, and numpy installed(obviously)
```angular2html
pip install numpy
pip install tensorflow keras
```

run the file to export the trained model
```angular2html
python index.py
```
#### Variables

There are 2 variables starting at line 15

**model_version**: change this to change the 
name of the folder of the specific model version
```angular2html
model_version = "1"
```
**epoch**: the higher this number is the more accurate the model, but the longer it will take to train. 5000 is good, but may take a while
```angular2html
epoch = 100
```

