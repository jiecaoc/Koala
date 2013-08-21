koala
========

<code>koala</code> allows you to rapidly train a random forest learner.  

####Requirements

- NumPy
- pandas
- copper

####Installation
    $ python setup.py install

####Bascis

	from koala import Koala
	import pandas as pd
	
	wine = pd.read_csv('../data/wine.csv')
	kl = Koala(data=wine.drop('Magnesium', axis=1), target='Type')
	kl.train(test_size=0.3)
	print(kl.confusion_matrix())
	
	    1   2   3
	1  23   0   0
	2   1  17   0
	3   0   0  13

More information and code samples available in the [notebooks](https://github.com/colindickson/Koala/tree/master/notebooks) folder.
