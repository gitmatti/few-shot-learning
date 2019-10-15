# Proto Net Fashion Data experiments

python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 1 --k-train 10 --n-train 1 --q-train 5 --small-dataset --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --small-dataset --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --small-dataset --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 5 --k-train 10 --n-train 5 --q-train 5 --small-dataset --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 5 --k-train 30 --n-train 5 --q-train 5 --small-dataset --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 5 --k-train 30 --n-train 5 --q-train 5 --small-dataset --gpu 0

#python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 1 --k-train 10 --n-train 1 --q-train 5 --small-dataset --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --small-dataset --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --small-dataset --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 5 --k-train 10 --n-train 5 --q-train 5 --small-dataset --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 5 --k-train 30 --n-train 5 --q-train 5 --small-dataset --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 5 --k-train 30 --n-train 5 --q-train 5 --small-dataset --validate --gpu 1

# the full dataset models should be pretrained to avoid high-dimensionality in the embedding space
python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 1 --k-train 10 --n-train 1 --q-train 5 --pretrained --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --pretrained --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --pretrained --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 5 --k-train 10 --n-train 5 --q-train 5 --pretrained --gpu 0

# not enough memory for these with --k-train 30 (would be batchsize 300)
python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 5 --pretrained --gpu 0
python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 5 --k-train 20 --n-train 5 --q-train 5 --pretrained --gpu 0

#python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 1 --k-train 10 --n-train 1 --q-train 5 --pretrained --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --pretrained --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 1 --k-train 30 --n-train 1 --q-train 5 --pretrained --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 2 --n-test 5 --k-train 10 --n-train 5 --q-train 5 --pretrained --validate --gpu 1

# not enough memory for these with --k-train 30 (would be batchsize 300)
#python -m experiments.few_shot_learning --dataset fashion --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 5 --pretrained --validate --gpu 1
#python -m experiments.few_shot_learning --dataset fashion --k-test 15 --n-test 5 --k-train 20 --n-train 5 --q-train 5 --pretrained --validate --gpu 1

