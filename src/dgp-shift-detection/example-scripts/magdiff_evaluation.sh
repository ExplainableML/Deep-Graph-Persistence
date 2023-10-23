for dataset in cifar10 mnist fashion-mnist;
do
for method in topological dnp_normalize dnp_var_normalize;
do
for shift in gaussian_noise gaussian_blur salt_pepper_noise image_transform pixel_shuffle pixel_dropout uniform_noise;
do
for intensity in 0 1 2 3 4 5;
do
sbatch --export=dataset=$dataset,shift=$shift,method=$method,intensity=$intensity magdiff_evaluation.batch
done;
done;
done;
done;
