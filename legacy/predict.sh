#!/bin/bash
cd  /Midgard/home/paulbp/xview2-replication/
python3 predict34_loc.py 
python3 predict50_loc.py
python3 predict92_loc.py
# python3 predict154_loc.py
python3 predict34cls.py 0 winner True
python3 predict34cls.py 1 winner True
python3 predict34cls.py 2 winner True
python3 predict50cls.py 0 winner True
python3 predict50cls.py 1 winner True
python3 predict50cls.py 2 winner True
python3 predict92cls.py 0 winner True
python3 predict92cls.py 1 winner True
python3 predict92cls.py 2 winner True
# python3 predict154cls.py 0 winner True
# python3 predict154cls.py 1 winner True
# python3 predict154cls.py 2 winner True
python3 create_submission.py
echo "submission created!"