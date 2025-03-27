##################################            NBA              ##################################

#python training0112.py --dataname nba  --a 8 --b 1 --epochs 1000 --patience 100 --wd 1e-3 --lr 0.0005 --wd1 1e-3 --lr1 0.002  --lr3 0.08 --wd3 1e-6 --is_bns False  --act_fn prelu  --hid_dim 128 --dprate 0.3 --dropout 0.0  --acc 0.70  --roc 0.72 --label_number 100 --test_idx True

python training0112.py --dataname nba  --a 8 --b 1 --epochs 1000 --patience 100 --wd 1e-3 --lr 0.0005 --wd1 1e-3 --lr1 0.002  --lr3 0.08  --is_bns False  --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.0  --acc 0.70  --f1 0.70 --label_number 100 --test_idx True





##################################            Pokec-z              ##################################

#python training0112.py --dataname pokec-z --a 8 --b 1  --epochs 1000 --seed 20 --patience 100 --wd 1e-5 --lr 0.0001 --wd1 0.001 --lr1 0.02  --lr3 0.08 --wd3 1e-5 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.3  --acc 0.65  --roc 0.69 --label_number 1000 --test_idx False
#python training0112.py --dataname pokec-z --a 8 --b 1  --epochs 1000 --seed 20 --patience 100 --wd 1e-5 --lr 0.0001 --wd1 0.001 --lr1 0.02  --lr3 0.008 --wd3 1e-5 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.3  --acc 0.65  --roc 0.69 --label_number 1000 --test_idx False

python training0112.py --dataname pokec-z --a 2 --b 1  --epochs 1000 --seed 20 --patience 100 --wd 1e-5 --lr 0.0001 --wd1 0.001 --lr1 0.02  --lr3 0.008 --wd3 1e-5 --is_bns True --act_fn prelu  --hid_dim 128 --dprate 0.3 --dropout 0.1  --acc 0.65  --f1 0.69 --label_number 1000 --test_idx T





##################################            Pokec-n              ##################################

#python training0112.py --a 1 --b 1 --dataname pokec-n --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns True --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.65  --roc 0.68 --label_number 1000 --test_idx False
#python training0112.py --a 2 --b 2 --dataname pokec-n --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.65  --roc 0.68 --label_number 1000 --test_idx False

python training0112.py --a 2 --b 1 --dataname pokec-n --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.3 --dropout 0.1  --acc 0.65  --f1 0.64 --label_number 1000  
#python training0112.py --a 5 --b 5 --dataname pokec-n --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.65  --roc 0.68 --label_number 1000 --test_idx False





##################################            German              ##################################

python training0112.py --a 2 --b 1 --dataname german --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.73  --f1 0.83 --label_number 600 --test_idx  T



##################################            Credit              ##################################

python training0112.py  --a 2 --b 1 --dataname credit --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.78  --f1 0.87 --label_number 1000  --test_idx  T



##################################            Bail              ##################################

python training0112.py --a 2 --b 1 --dataname bail --epochs 1000 --seed 20 --patience 100 --wd 1e-4 --lr 0.00005 --wd1 0.001 --lr1 0.002  --lr3 0.08 --wd3 1e-3 --is_bns False --act_fn prelu  --hid_dim 128 --dprate 0.1 --dropout 0.1  --acc 0.88  --f1 0.83 --label_number 1000  --test_idx  T



