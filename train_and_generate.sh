for i in $(seq 1 20)
do
  python train_or_generate.py && python train_or_generate.py --mode generate
done
