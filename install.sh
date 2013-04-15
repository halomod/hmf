sudo rm dist/*
sudo python setup.py sdist

cd dist
tar zxvf *.tar.gz
cd hmf*
sudo python setup.py install
cd ../..
#sudo rm -rf dist/hmf*/*
#rmdir dist/hmf*
#cd ..
rsync -a dist/hmf* smurray@pleiades.icrar.org: