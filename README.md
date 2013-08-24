koala
========

<code>koala</code> allows you to rapidly train a random forest classifier.  Useful for quick prototyping.

####Requirements

- Python 3.3+
- NumPy
- scikit-learn
- pandas
- copper

####Installation
    $ python setup.py install

####Basics

	from koala import Koala
	import pandas as pd
	
	wine = pd.read_csv('../data/wine.csv')
	kl = Koala(data=wine, target='Type')
	kl.train(test_size=0.3)
	print(kl.confusion_matrix())
	
	    1   2   3
	1  23   0   0
	2   1  17   0
	3   0   0  13

More information and code samples available in the [notebooks](https://github.com/colindickson/Koala/tree/master/notebooks) folder.
