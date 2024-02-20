
# test01

python ARCompassStage1.py -i data/test01.Stage1.fasta -o result/test01/ -p 0.9

python ARCompassStage2.py -i data/test01.Stage2.fasta -o result/test01/ -p 0.9

python ARCompassStage3.py -i data/test01.Stage3.fasta -o result/test01/


# test02

python ARCompassStage1.py -i data/test.1to100.fasta -o result/test02/ -p 0.99

python ARCompassStage2.py -i result/test02/Stage1.ARG.fasta -o result/test02/ -p 0.9

python ARCompassStage3.py -i result/test02/Stage2.beta_lactam.fasta -o result/test02/

