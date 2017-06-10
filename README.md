# keras-ppap
Pen Pineapple Apple Pen

"Building powerful image classification models using very little data"
by Yoshi Sakai as Bluemooninc. https://github.com/bluemooninc
It uses data that can be downloaded at:
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created apple/ pineapple and pen subfolders inside train/ and validation/
- put the apple,pineapple,pen pictures index 1-24 in data/train/ each folder
- put the dpple,pineapple,pen pictures index 25-30 in data/validation/ each folder
So that we have 24 training examples for each class, and 6 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        apple/
            apple001.jpg
            apple002.jpg
            ...
        pineapple/
            pineapple001.jpg
            pineapple002.jpg
            ...
        pen/
            pen001.jpg
            pen002.jpg
            ...
    validation/
        apple/
            apple025.jpg
            apple026.jpg
            ...
        pineapple/
            pineapple025.jpg
            pineapple026.jpg
            ...
        pen/
            pen025.jpg
            pen026.jpg
            ...
'''
