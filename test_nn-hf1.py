#!/usr/bin/python
import unittest
from tn import *

class TestTn(unittest.TestCase):

  def test_files_all(self):
      all_input = len(myfiles(PATH_PREFIX,""))
      num = len(myfiles(PATH_PREFIX,"num"))
      lower = len(myfiles(PATH_PREFIX,"lower"))
      upper = len(myfiles(PATH_PREFIX,"upper"))
      self.assertEqual(all_input,8571)
      self.assertEqual(num,600)
      self.assertEqual(lower,1470)
      self.assertEqual(upper,6501)
      self.assertEqual(all_input - lower - upper -num, 0)

  def test_maxlength(self):
      self.assertEqual(max_length(myfiles(PATH_PREFIX,"num")), 218)
      self.assertEqual(max_length(myfiles(PATH_PREFIX,"lower")), 163)
      self.assertEqual(max_length(myfiles(PATH_PREFIX,"upper")), 412)

  def test_extend_to(self):
      inputs = collections.defaultdict(dict)
      inputs["a"]["b"] = dict(x=[8.0,7.0,3.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      result = extendTo(inputs, 5)
      a = inputs["a"]["b"]["all"]
      # extended with zeros
      self.assertEqual(sum(a[3:5]) + sum(a[8:10]) + sum(a[13:15]), 0) 
      # new length
      self.assertEqual(len(a), 15)

  def test_normalize(self):
      self.assertEqual(normalize(float(4),float(0),float(10)), 0.4)
      self.assertEqual(normalize(float(4),float(0),float(5)), 0.8)

  def test_create_dataset(self):
      inputs = collections.defaultdict(dict)
      inputs["a"]["a"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["b"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["c"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["d"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["e"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["f"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["g"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["h"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["i"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      inputs["a"]["j"] = dict(x=[3.0,4.0,5.0], y=[2.0,4.0,5.0], z=[3.0,5.0,4.0])
      extended = extendTo(inputs, 5)
      result = create_dataset(extended)
      print result[0][1]
      self.assertEqual(len(result[0][0]), 7)
      self.assertEqual(len(result[0][0][0]), 15)
      self.assertEqual(len(result[1][0]), 2)
      self.assertEqual(len(result[2][0]), 1)
      self.assertEqual(len(result[0][1]), 7)
      self.assertEqual(len(result[1][1]), 2)
      self.assertEqual(len(result[2][1]), 1)

if __name__ == '__main__':
    unittest.main()
