set -e

echo "Install Pandarallel"
echo "-------------------"
echo
pip install -e .[dev] --quiet

echo "Pandas 1.3.0"
echo "------------"
echo
pip install pandas==1.3.0 --quiet
pytest

echo "Pandas 1.4.0"
echo "------------"
echo
pip install pandas==1.4.0 --quiet
pytest

echo "Pandas latest"
echo "------------"
echo
pip install pandas --quiet --upgrade
pytest