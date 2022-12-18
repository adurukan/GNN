python3.10 -m venv ../.venvs/GNN
source ../.venvs/GNN/bin/activate
python3.10 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
python3.10 -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install pandas
pip install dvc
dvc init
dvc add data
git add data.dvc .gitignore 
git commit -m "Add raw data"
dvc remote add -d my_local_remote .dvc/tmp/dvcstore
# To test dvc pull
rm -rf .dvc/cache
rm -f data/data.xml
dvc pull
dvc repro -> to run the stages.

