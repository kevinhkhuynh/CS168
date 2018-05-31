apt-get -y install python2.7
apt-get -y install python-dev
pip install virtualenv
virtualenv -p /usr/bin/python2.7 Vpy27
source Vpy27/bin/activate
pip install tensorflow==0.12.0
pip install scikit-image
pip install pandas