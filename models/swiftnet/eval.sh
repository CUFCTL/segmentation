#!/bin/sh

#python eval.py configs/rn18_pyramid.py live
#python eval.py configs/rn18_pyramid.py static
#python eval.py configs/rn18_pyramid_rellis.py static
python eval.py --timing configs/rn18_pyramid_rellis.py static
#python eval.py --timing configs/rn18_pyramid.py static 
