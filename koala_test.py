import os
import platform
import numpy as np
import pandas as pd
import koala
import unittest

if (platform.uname())[0] == "Windows":
    FOLDER = os.environ['USERPROFILE']
else:
    FOLDER = os.environ.get("HOME")
FILE = 'test.dat'
PATH = os.path.join(FOLDER, FILE)

DATASET = pd.read_csv('data/iris.csv')

class KoalaTest(unittest.TestCase):

    def testKoalaInstantiation(self):
        try:
            k = koala.Koala()

        except Exception as e:
            self.fail(str(e))


    def testKoalaSetData(self):
        try:
            k = koala.Koala()
            k.set_data(DATASET)
            self.assertEqual(k.data.frame, DATASET)

        except Exception as e:
            self.fail(str(e))

    def testKoalaSaveAndLoad(self):
        try:
            k = koala.Koala()
            k.set_data(DATASET)

            k.set_target(4)
            k.train(test_size=0.05)
            k.save(PATH)

            l = koala.Koala()
            l.load(PATH)

            self.assertEqual(k.data, l.data)
            self.assertEqual(k._mc, l._mc)
        except Exception as e:
            self.fail(str(e))
        else:
            os.remove(PATH)

    def testKoalaClassification(self):
        try:
            X_test = np.array([[5,3,1,0],[8,3,6,2]])
            k = koala.Koala(data=DATASET,target='species')
            k.train(test_size=0.4)
            predictions = k.predict(X_test)

            self.assertEqual(predictions[0].lower(), 'iris-setosa')
            self.assertEqual(predictions[1].lower(), 'iris-virginica')

        except Exception as e:
            self.fail(str(e))

    def testKoalaClassiferFeatures(self):
        try:
            k = koala.Koala(data=DATASET, target='species')
            k.train()
            fs = k.feature_importance()
            fr = k.feature_reduction_scores()
        except Exception as e:
            self.fail(str(e))

if __name__ == "__main__":
    unittest.main()
